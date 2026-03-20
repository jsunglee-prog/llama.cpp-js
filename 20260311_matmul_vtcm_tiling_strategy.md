# Hexagon DSP: MatMul VTCM 타일링 전략 분석

> 작성일: 2026-03-11  
> 관련 파일: `ggml/src/ggml-hexagon/htp/matmul-ops.c`, `flash-attn-ops.c`

---

## 1. 문제 요약

### 1.1 Flash Attention vs MatMul의 VTCM 사용 차이

LLM 추론 파이프라인에서 Hexagon DSP로 오프로드되는 연산은 크게 두 종류입니다:

| 연산 | VTCM 사용 방식 | 시퀀스/토큰 수 증가 시 |
|------|---------------|---------------------|
| **Flash Attention** | K/V를 64행 블록으로 분할, 더블 버퍼링 | **VTCM 사용량 일정** (자동 타일링) |
| **MatMul** (QKV Proj, FFN 등) | src1 전체를 VTCM에 로딩 | **VTCM 사용량 선형 증가** → 초과 가능 |

Flash Attention은 내부적으로 `FLASH_ATTN_BLOCK_SIZE=64`로 K/V를 자동 분할하므로, 시퀀스 길이에 관계없이 VTCM이 부족해지지 않습니다.

그러나 **Attention을 제외한 나머지 모든 Linear 연산(MatMul)**은 src1(입력 토큰)을 VTCM에 통째로 올려야 하므로, 토큰 수가 커지면 VTCM 8MB를 초과합니다.

### 1.2 LLM 파이프라인에서 FA가 커버하는 범위

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer Layer                            │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ Q Projection │   │ K Projection │   │ V Projection │        │
│  │  (MatMul)    │   │  (MatMul)    │   │  (MatMul)    │        │
│  │  ⚠ 타일링필요 │   │  ⚠ 타일링필요 │   │  ⚠ 타일링필요 │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                   │                │
│         ▼                  ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Flash Attention                        │       │
│  │         ✅ 자동 64행 블록 타일링                      │       │
│  │         시퀀스 길이 무관, VTCM 사용량 일정             │       │
│  └────────────────────────┬────────────────────────────┘       │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────┐                      │
│  │        Output Projection (MatMul)    │                      │
│  │        ⚠ 타일링 필요                  │                      │
│  └──────────────────────────┬───────────┘                      │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────┐                      │
│  │           FFN (MatMul × 2~3)         │                      │
│  │     up_proj / gate_proj / down_proj  │                      │
│  │        ⚠ 타일링 필요                  │                      │
│  └──────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘

✅ = FA가 커버 (타일링 불필요)
⚠  = MatMul, src1이 VTCM을 초과하면 문제 발생
```

**결론**: FA는 Attention 연산 하나만 커버합니다. QKV Projection, Output Projection, FFN의 MatMul들은 토큰 수가 클 때 별도 대응이 필요합니다.

---

## 2. 현재 MatMul 코드의 경로 분석

### 2.1 경로 선택 흐름

> 코드 위치: `matmul-ops.c` `op_matmul()` (line 2334~)

```
op_matmul() 진입
  │
  ├── src0 == F16?
  │     │
  │     ├── src1 전체 VTCM에 들어감? (f16_total_size ≤ vtcm_size)
  │     │     └── ✅ matmul_2d (최적: src1 VTCM + src0 DMA 스트리밍)
  │     │
  │     └── src1 VTCM 초과
  │           └── ⚠️ matmul_4d (폴백: DDR 직접, DMA 없음, 느림)
  │
  └── src0 == Q4_0 / Q8_0 / MXFP4?
        │
        ├── src1 양자화(Q8x4x2) 후 VTCM에 들어감?
        │     └── ✅ matmul_2d / matvec_2d (최적)
        │
        └── src1 양자화 후에도 VTCM 초과
              └── ❌ HTP_STATUS_VTCM_TOO_SMALL (에러 반환!)
                    → 폴백 경로 없음!
```

### 2.2 핵심 문제: 양자화 경로에 폴백이 없음

F16 경로는 src1이 VTCM에 안 들어가면 `matmul_4d`로 폴백하지만, **양자화 경로(Q4_0/Q8_0/MXFP4)에서는 에러를 반환**합니다.

> 코드 위치: `matmul-ops.c` line 2443~2446

```c
if (octx->ctx->vtcm_size < spad_size) {
    FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n",
         mmctx->type, octx->ctx->vtcm_size, spad_size);
    return HTP_STATUS_VTCM_TOO_SMALL;
}
```

### 2.3 VTCM 초과가 발생하는 입력 크기 예시

6스레드 기준, `ne[0]`(feature dim)과 `src1_nrows`(토큰 수)에 따른 VTCM 요구량:

| ne[0] | src1_nrows | src1 타입 | src1 VTCM (양자화 후) | src0+dst spad | 총합 | 8MB 초과? |
|-------|-----------|----------|---------------------|--------------|------|----------|
| 784 | 2048 | Q8x4x2 | ~1.75 MB | ~0.1 MB | ~1.85 MB | ✓ 적합 |
| 4096 | 512 | Q8x4x2 | ~2.1 MB | ~0.2 MB | ~2.3 MB | ✓ 적합 |
| 4096 | 2048 | Q8x4x2 | ~8.5 MB | ~0.2 MB | ~8.7 MB | **✗ 초과** |
| 4096 | 4096 | Q8x4x2 | ~17 MB | ~0.2 MB | ~17.2 MB | **✗ 초과** |
| 11008 | 2048 | Q8x4x2 | ~23 MB | ~0.5 MB | ~23.5 MB | **✗ 초과** |

**실제 LLM에서**: LLaMA-7B의 FFN intermediate_size=11008, LLaMA-13B는 13824. Prompt processing에서 토큰 수가 수백 이상이면 거의 확실히 VTCM을 초과합니다.

---

## 3. 대응 전략

### 3.1 방법 1: src1 행 분할 타일링 (권장, `op_matmul` 수정)

**원리**: src1의 행(토큰) 축을 VTCM에 들어갈 크기로 분할하여 반복 처리

```
원래: src1 전체 [K, 2048] → VTCM (초과!)

변경:
  chunk_rows = VTCM_available / q8x4x2_row_size(K)

  ┌──────────────────────────────────────────────────┐
  │ 반복 1: src1[0 : chunk_rows]                     │
  │   1. 양자화(F32→Q8x4x2) → VTCM                  │
  │   2. src0 전체를 16행씩 DMA 스트리밍              │
  │   3. dst[0 : chunk_rows] 열에 결과 기록           │
  ├──────────────────────────────────────────────────┤
  │ 반복 2: src1[chunk_rows : 2*chunk_rows]          │
  │   1. 양자화(F32→Q8x4x2) → VTCM                  │
  │   2. src0 전체를 16행씩 DMA 스트리밍              │
  │   3. dst[chunk_rows : 2*chunk_rows] 열에 결과 기록│
  ├──────────────────────────────────────────────────┤
  │ ... 반복 (모든 토큰 처리 완료까지)                 │
  └──────────────────────────────────────────────────┘
```

**정확성 보장**: 행렬곱 `C = A × B`에서, B의 열(행) 분할은 C의 대응 열(행)에만 영향을 미치므로, 청크 간 의존성이 없어 분할 결과가 수학적으로 동일합니다.

**수정 위치**: `op_matmul()` 함수 내부

```c
// 현재 코드 (matmul-ops.c line 2443)
if (octx->ctx->vtcm_size < spad_size) {
    return HTP_STATUS_VTCM_TOO_SMALL;
}

// 개선 방향 (의사코드):
if (octx->ctx->vtcm_size < spad_size) {
    // src1을 청크로 분할
    size_t vtcm_for_src1 = octx->ctx->vtcm_size - (src0_spad_size + dst_spad_size);
    uint32_t chunk_rows = vtcm_for_src1 / src1_row_size;
    chunk_rows = (chunk_rows / 2) * 2;  // 2의 배수로 정렬 (2x2 tiling)

    for (uint32_t chunk_start = 0; chunk_start < src1_nrows; chunk_start += chunk_rows) {
        uint32_t current_chunk = MIN(chunk_rows, src1_nrows - chunk_start);

        // 1. src1 청크만 양자화 → VTCM
        // 2. matmul_2d 실행 (src1_nrows = current_chunk)
        // 3. dst offset 조정
    }
    return HTP_STATUS_OK;
}
```

**장점**:
- 기존 `matmul_2d()` 내부 로직 거의 수정 없이, 바깥에서 루프만 감싸면 됨
- DMA 파이프라이닝(src0 스트리밍) 완전 유지
- HVX 2×2 벡터 타일링 완전 유지

**단점**:
- src0(가중치) 전체를 청크 수만큼 반복 순회 → src0가 클수록 I/O 증가
- 양자화도 청크마다 반복 실행

**성능 영향 추정**:

```
단일 패스 (src1 전체 VTCM):
  DMA: src0 전체 1회 스트리밍
  연산: src0_rows × src1_rows 내적

2청크 분할 시:
  DMA: src0 전체 2회 스트리밍 (오버헤드)
  연산: src0_rows × (src1_rows/2) 내적 × 2 = 동일
  양자화: src1_rows/2 × 2 = 동일

→ 연산량은 동일, DMA 오버헤드는 청크 수에 비례하여 증가
→ 2~4 청크 정도면 DMA 오버헤드는 ~10-20% 수준
```

### 3.2 방법 2: `matmul_4d` 폴백 확장 (에러 방지용)

현재 F16 경로만 `matmul_4d` 폴백이 있는데, 양자화 경로에도 적용:

> 코드 위치: `matmul-ops.c` line 2393~2417 (F16 폴백 경로)

```c
// 현재 (F16만 해당):
} else {
    // Fallback to f16/f32 (DDR) if src1 doesn't fit in VTCM
    matmul_job_func = matmul_4d;
    // ...
}
```

양자화 경로에도 유사한 폴백 추가:

```c
// 양자화 경로 (의사코드):
if (octx->ctx->vtcm_size < spad_size) {
    // matmul_4d 폴백 (DDR 직접 접근, 양자화 없이 원본 타입 사용)
    mmctx->vec_dot_1x1 = vec_dot_q4_0_q8_0_1x1;  // DDR용 vec_dot
    matmul_job_func = matmul_4d;
    need_quant = false;
    // ... fastdiv 초기화 ...
}
```

**장점**: 간단히 구현 가능, 에러 대신 동작은 함
**단점**: VTCM + DMA를 전혀 못 쓰므로 **성능 3~5배 저하**

```
matmul_4d의 동작 방식:
  - VTCM 사용하지 않음
  - DDR에서 직접 데이터 읽기
  - L2 캐시에 의존
  - 64×64 블록 루프로 캐시 재사용만 향상
  - DMA 파이프라이닝 없음
  - HVX 정렬 보장 안 됨 (unaligned vec_dot 사용)
```

### 3.3 방법 3: 상위 레벨 배치 크기 제한 (코드 수정 없음)

llama.cpp의 배치 파라미터로 한 번에 처리하는 토큰 수를 제한:

```bash
# ubatch(micro-batch)를 줄여서 MatMul src1이 VTCM에 들어오도록 조절
llama-cli -m model.gguf -ub 256

# 예시: ne[0]=4096인 모델에서 VTCM 버짓 계산
# q8x4x2_row_size(4096) ≈ 4352 B
# 사용 가능 VTCM ≈ 7.5 MB (src0/dst spad 제외)
# max_rows = 7.5MB / 4352 ≈ 1808행
# → -ub 1808 이하로 설정하면 VTCM 초과 방지
```

**모델별 권장 ubatch 상한**:

| 모델 | ne[0] (hidden_dim) | q8x4x2 행 크기 | 최대 ubatch (≈) |
|------|-------------------|----------------|----------------|
| TinyLLaMA | 2048 | ~2176 B | ~3600 |
| LLaMA-7B | 4096 | ~4352 B | ~1800 |
| LLaMA-13B | 5120 | ~5376 B | ~1460 |
| LLaMA-7B FFN | 11008 | ~11264 B | ~700 |
| LLaMA-13B FFN | 13824 | ~14080 B | ~560 |

**주의**: FFN의 intermediate_size가 hidden_dim보다 훨씬 크므로 (보통 2.7~4배), FFN이 병목이 됩니다. 가장 큰 ne[0] 기준으로 ubatch를 설정해야 합니다.

**장점**: 코드 수정 없음, 즉시 적용 가능
**단점**: prompt processing 처리량(throughput) 감소, 모델마다 튜닝 필요

---

## 4. 전략 비교 및 권장 사항

| | 방법 1: src1 분할 타일링 | 방법 2: matmul_4d 폴백 | 방법 3: ubatch 제한 |
|---|---|---|---|
| **구현 난이도** | 중 | 하 | 없음 |
| **성능 영향** | ~10-20% 오버헤드 (DMA 반복) | ~3-5× 저하 | throughput 감소 |
| **DMA 파이프라이닝** | ✅ 유지 | ❌ 없음 | ✅ 유지 |
| **HVX 정렬 최적화** | ✅ 유지 | ❌ 미정렬 접근 | ✅ 유지 |
| **코드 수정 범위** | `op_matmul()` | `op_matmul()` + vec_dot | 없음 |
| **임의 크기 대응** | ✅ 자동 | ✅ 자동 | ❌ 수동 튜닝 |
| **적용 시점** | 장기 | 중기 (안전망) | 즉시 |

### 권장 로드맵

```
즉시 (단기)
  └── 방법 3: -ub 파라미터로 VTCM 초과 방지
        → 모델별 최적 ubatch 프로파일링

1-2주 (중기)
  └── 방법 2: matmul_4d 폴백 추가
        → 양자화 경로에서 에러 대신 느리게라도 동작
        → 예상치 못한 크기 입력에 대한 안전망

2-4주 (장기)
  └── 방법 1: src1 행 분할 타일링 구현
        → op_matmul()에 청크 루프 추가
        → 기존 matmul_2d() 재사용
        → 최적 성능 유지하면서 임의 크기 지원
```

---

## 5. 방법 1 구현 시 상세 설계

### 5.1 수정 대상 함수

`op_matmul()` — `matmul-ops.c` line 2334

### 5.2 변경할 로직 흐름

```
op_matmul() {
    // ... 기존 초기화 ...

    spad_size = src1_spad + src0_spad + dst_spad;

    if (vtcm_size >= spad_size) {
        // 기존 경로: 단일 패스
        양자화() → matmul_2d/matvec_2d()
    } else {
        // 신규 경로: src1 청크 분할
        vtcm_for_src1 = vtcm_size - (src0_spad + dst_spad);
        chunk_rows = vtcm_for_src1 / src1_row_size;
        chunk_rows = ALIGN_DOWN(chunk_rows, 2);  // 2x2 tiling 호환

        for (chunk = 0; chunk < src1_nrows; chunk += chunk_rows) {
            current_n = MIN(chunk_rows, src1_nrows - chunk);

            // src1 spad 크기 재설정
            src1_spad.size = current_n * src1_row_size;
            // VTCM 재배치
            recompute_vtcm_layout();

            // 양자화: src1의 현재 청크만
            quantize_chunk(src1, chunk, current_n);

            // matmul: src0 전체 × src1 청크
            // dst의 오프셋 조정 필요
            mmctx->chunk_offset = chunk;
            mmctx->chunk_nrows  = current_n;
            matmul_2d() or matvec_2d();
        }
    }
}
```

### 5.3 `matmul_2d` 내부 수정 사항

현재 `matmul_2d()`는 dst에 쓸 때 src1 인덱스를 그대로 사용합니다:

```c
// 현재 (matmul-ops.c line 1625-1630)
for (uint32_t ir1 = 0; ir1 + 1 < src1_nrows; ir1 += 2) {
    float * dst_row0 = (float *)(dst->data + ((ir1+0) * dst_row_size));
    float * dst_row1 = (float *)(dst->data + ((ir1+1) * dst_row_size));
    // ...
}
```

청크 분할 시, `ir1`에 `chunk_offset`을 더해야 올바른 dst 위치에 기록됩니다:

```c
// 변경 후
const uint32_t chunk_offset = mmctx->chunk_offset;
for (uint32_t ir1 = 0; ir1 + 1 < src1_nrows; ir1 += 2) {
    float * dst_row0 = (float *)(dst->data + ((chunk_offset + ir1 + 0) * dst_row_size));
    float * dst_row1 = (float *)(dst->data + ((chunk_offset + ir1 + 1) * dst_row_size));
    // ...
}
```

### 5.4 양자화 함수 수정 사항

현재 `quantize_f32_q8x4x2()`는 src1 전체를 양자화합니다. 청크 분할 시, 시작 오프셋과 행 수를 파라미터로 받아야 합니다.

```c
// 현재: src1 전체를 양자화하여 VTCM에 저장
// 변경: src1[chunk_start : chunk_start + chunk_rows]만 양자화
```

### 5.5 Flash Attention과의 구조 비교

| | Flash Attention (기존) | MatMul src1 분할 (신규) |
|---|---|---|
| **분할 대상** | K/V의 시퀀스 축 | src1의 행(토큰) 축 |
| **블록 크기** | 고정 64행 | 동적 (VTCM 여유 기반) |
| **버퍼링** | 더블 버퍼 (A/B 교대) | 싱글 버퍼 (청크 교체) |
| **누적 방식** | Online Softmax (점진적) | 독립 (청크 간 의존 없음) |
| **DMA 파이프라인** | ib+2 프리페치 | src0 16행 프리페치 (기존 유지) |

MatMul의 경우 청크 간 의존성이 없으므로 FA의 online softmax 같은 복잡한 누적 로직이 불필요합니다. 각 청크의 결과가 dst의 서로 다른 행에 독립적으로 기록됩니다.

---

## 6. 요약

1. **Flash Attention**은 K/V를 자동으로 64행 블록 타일링하므로, 시퀀스 길이에 상관없이 VTCM 문제가 없습니다.
2. **MatMul**(QKV Projection, FFN 등)은 src1 전체를 VTCM에 올려야 하므로, 토큰 수가 크면 VTCM을 초과합니다.
3. **현재 양자화 경로(Q4_0/Q8_0/MXFP4)에는 폴백이 없어**, VTCM 초과 시 에러를 반환합니다.
4. **단기 대응**: `-ub` 파라미터로 배치 크기를 제한 (코드 수정 없음)
5. **장기 대응**: `op_matmul()`에 src1 행 분할 타일링을 구현하여, 기존 `matmul_2d()`의 DMA 파이프라이닝을 유지하면서 임의 크기를 자동 지원
