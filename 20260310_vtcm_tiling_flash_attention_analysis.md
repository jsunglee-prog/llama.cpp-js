# Hexagon DSP: VTCM 타일링 및 Flash Attention 분석

> 작성일: 2026-03-10
> 분석 대상: `ggml/src/ggml-hexagon/` 전체, 특히 `htp/matmul-ops.c`, `htp/flash-attn-ops.c`, `htp/main.c`

---

## 1. 문제 정의: [784, 2048] 입력이 VTCM을 초과하는 경우

### 1.1 VTCM 용량

VTCM(Vector Tightly Coupled Memory)은 Hexagon DSP의 고속 온칩 메모리로, 기본 **8MB**입니다.

> 코드 위치: [htp/main.c:203](ggml/src/ggml-hexagon/htp/main.c)
```c
unsigned int vtcm_size = 8 * 1024 * 1024;  // 8MB default
HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);
```

### 1.2 [784, 2048] 입력의 메모리 요구량

`[784, 2048]` 텐서의 실제 메모리 사이즈를 데이터 타입별로 계산하면:

| 데이터 타입 | 행 크기 (784 elements) | 전체 크기 (784 × 2048 rows) | VTCM 8MB 대비 |
|-------------|----------------------|---------------------------|---------------|
| **F32** | 3,136 B | **6,291,456 B (6.0 MB)** | 75% |
| **F16** | 1,568 B | **3,145,728 B (3.0 MB)** | 37.5% |
| **Q8_0** (q8x4x2 repack) | ~816 B | **1,671,168 B (~1.6 MB)** | ~20% |
| **Q4_0** (q4x4x2 repack) | ~432 B | **884,736 B (~0.86 MB)** | ~11% |

**핵심**: F32 기준 6MB 자체는 8MB VTCM에 들어가지만, 실제로는 src0(가중치) 스크래치패드, dst 스크래치패드, 멀티스레드 복제 공간까지 합산해야 하므로 **VTCM 총 요구량이 8MB를 쉽게 초과**합니다.

### 1.3 실제 VTCM 요구량 계산 (MatMul 기준)

MatMul `op_matmul()`에서 src1이 `[784, 2048]`이고 F32 타입일 때, VTCM 스크래치패드 구성:

> 코드 위치: [htp/matmul-ops.c:2316-2340](ggml/src/ggml-hexagon/htp/matmul-ops.c)

```
src1_spad: q8x4x2_row_size(784) × 2048 rows  = ~1.6 MB (양자화 후)
src0_spad: MM_SPAD_SRC0_NROWS(16) × row_size × n_threads(6)
dst_spad:  MM_SPAD_DST_NROWS(2) × row_size × n_threads(6)
```

**src1 전체를 VTCM에 올려야 하는 것이 핵심 제약**입니다. `matmul_2d()` 경로에서는 src1이 미리 양자화(F32→Q8x4x2)되어 VTCM에 전부 로딩된 상태에서 src0를 16행 단위로 DMA 스트리밍합니다.

---

## 2. 현재 Hexagon 백엔드의 타일링 전략

### 2.1 MatMul에서의 타일링 구조

현재 구현은 **세 가지 경로**를 사용합니다:

#### 경로 1: `matmul_2d()` — 주 경로 (src1이 VTCM에 들어갈 때)

> 코드 위치: [htp/matmul-ops.c:1569](ggml/src/ggml-hexagon/htp/matmul-ops.c)

**전제 조건**: src1 전체가 VTCM에 로딩되어야 함

```
VTCM 레이아웃:
┌──────────────────────────────────────────────────────────────────────┐
│ src0_spad (per-thread)                                               │
│   = MM_SPAD_SRC0_NROWS(16) × src0_row_size_padded × n_threads       │
│   → src0(가중치) 16행을 DMA로 순환 프리페치                            │
├──────────────────────────────────────────────────────────────────────┤
│ src1_spad (shared, 전체!)                                            │
│   = src1_nrows × src1_row_size                                       │
│   → 입력 텐서 전체가 양자화(Q8x4x2) 후 VTCM에 상주                    │
├──────────────────────────────────────────────────────────────────────┤
│ dst_spad (per-thread)                                                │
│   = MM_SPAD_DST_NROWS(2) × dst_row_size × n_threads                 │
└──────────────────────────────────────────────────────────────────────┘
```

**DMA 파이프라이닝 패턴**:

```
1. 사전 프리페치: src0의 처음 16행을 VTCM으로 DMA 전송
2. 메인 루프:
   for (ir0 = start; ir0 < end; ir0 += 2):
     ss0 = dma_queue_pop()              ← 현재 2행 DMA 완료 대기
     for (ir1 = 0; ir1 < src1_nrows; ir1 += 2):
       vec_dot_2x2(src0[ir0], src0[ir0+1], src1[ir1], src1[ir1+1])
     dma_queue_push(다음 2행)           ← 비동기 프리페치
```

> 코드 위치: [htp/matmul-ops.c:1620-1660](ggml/src/ggml-hexagon/htp/matmul-ops.c)

```c
// Prefill spad with src0 rows
for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
    const int is0 = (ir0 - src0_start_row);
    if (is0 >= MM_SPAD_SRC0_NROWS) break;
    dma_queue_push_ddr_to_vtcm(dma_queue, ...);
}

// Process src0 rows
for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
    const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;  // DMA 완료 대기

    // 2×2 타일 연산
    for (; ir1 + 1 < src1_nrows; ir1 += 2)
        mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ...);

    // 다음 블록 프리페치
    const int pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
    if (pr0 < src0_end_row_x2) {
        dma_queue_push_ddr_to_vtcm(dma_queue, ...);  // 비동기
    }
}
```

#### 경로 2: `matmul_4d()` — 폴백 경로 (src1이 VTCM에 안 들어갈 때)

> 코드 위치: [htp/matmul-ops.c:1500](ggml/src/ggml-hexagon/htp/matmul-ops.c)

**VTCM 타일링 없이** DRAM에서 직접 데이터를 읽고 블록 타일링(blck_0=64, blck_1=64)만 적용합니다.

```c
const uint32_t blck_0 = 64;
const uint32_t blck_1 = 64;

for (iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1)
    for (iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0)
        for (ir1 = iir1; ...)
            for (ir0 = iir0; ...)
                mmctx->vec_dot_1x1(ne00, &dst_col[ir0], src0_row, src1_col);
```

이 경로는 DMA 파이프라이닝이 없고, L2 캐시에 의존합니다. **성능이 상당히 떨어집니다.**

#### 경로 선택 로직

> 코드 위치: [htp/matmul-ops.c:2370-2420](ggml/src/ggml-hexagon/htp/matmul-ops.c)

```c
// F16 src0 경로
if (!is_batched && !is_permuted && f16_total_size <= octx->ctx->vtcm_size) {
    // 최적화 경로: matmul_2d (src1 전체 VTCM 로딩)
} else {
    // 폴백: matmul_4d (DRAM 직접 접근, DMA 없음)
}

// 양자화 src0 경로 (Q4_0, Q8_0, MXFP4)
// → 항상 matmul_2d 또는 matvec_2d 사용
// → src1이 VTCM에 안 들어가면 HTP_STATUS_VTCM_TOO_SMALL 에러 반환!
```

### 2.2 핵심 문제: 양자화 경로에서 src1이 VTCM을 초과하면 에러 발생

> 코드 위치: [htp/matmul-ops.c:2443-2446](ggml/src/ggml-hexagon/htp/matmul-ops.c)

```c
if (octx->ctx->vtcm_size < spad_size) {
    FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n",
         mmctx->type, octx->ctx->vtcm_size, spad_size);
    return HTP_STATUS_VTCM_TOO_SMALL;
}
```

**현재 양자화 경로(Q4_0/Q8_0/MXFP4)에는 src1 분할 타일링이 구현되어 있지 않습니다.** src1이 VTCM에 맞지 않으면 그냥 에러를 반환합니다. F16 경로만 `matmul_4d()` 폴백이 있습니다.

### 2.3 다른 연산들의 VTCM 타일링 패턴

#### Binary Ops (ADD, MUL, SUB, DIV)

> 코드 위치: [htp/binary-ops.c:775-810](ggml/src/ggml-hexagon/htp/binary-ops.c)

```c
size_t rows_per_buffer = octx->ctx->vtcm_size / (n_threads * spad_row_total);
// → VTCM에 들어가는 행 수를 계산하여 동적으로 타일 크기 결정
```

Binary Op들은 **행 수를 동적으로 조절하여 VTCM에 맞춤** — 한 번에 처리할 행 수를 VTCM 크기에 따라 줄입니다.

#### Activation Ops (SILU, GELU, SWIGLU 등)

> 코드 위치: [htp/act-ops.c:715-720](ggml/src/ggml-hexagon/htp/act-ops.c)

```c
size_t vtcm_row_per_thread = (octx->ctx->vtcm_size) / (n_threads * spad_size_per_row);
// → 스레드당 처리할 행 수를 동적으로 계산
```

---

## 3. Flash Attention의 VTCM 타일링 분석

### 3.1 활성화 방법

Flash Attention은 **experimental** 기능으로, 환경변수로 활성화해야 합니다:

```bash
export GGML_HEXAGON_EXPERIMENTAL=1
```

> 코드 위치: [ggml-hexagon.cpp:47](ggml/src/ggml-hexagon/ggml-hexagon.cpp), [ggml-hexagon.cpp:1780](ggml/src/ggml-hexagon/ggml-hexagon.cpp)

```cpp
static int opt_experimental = 0;

static bool ggml_hexagon_supported_flash_attn_ext(...) {
    // ... 타입 체크 ...
    return opt_experimental;  // experimental 플래그가 켜져야 지원
}
```

### 3.2 Flash Attention의 VTCM 레이아웃

> 코드 위치: [htp/flash-attn-ops.c:661-710](ggml/src/ggml-hexagon/htp/flash-attn-ops.c)

```
VTCM 구성 (스레드별):
┌───────────────────────────────────────────────────────┐
│ spad_q: Q 1행                                         │
│   = size_q_row_padded × 1                             │
│   예: DK=128, F16 → 256 B (128B 정렬)                 │
├───────────────────────────────────────────────────────┤
│ spad_k: K 블록 × 2 (더블 버퍼!)                        │
│   = size_k_row_padded × FLASH_ATTN_BLOCK_SIZE(64) × 2 │
│   예: DK=128 → 128×2B=256B/row, 패딩→256B, ×64×2     │
│   = 32,768 B (32 KB) per thread                       │
├───────────────────────────────────────────────────────┤
│ spad_v: V 블록 × 2 (더블 버퍼!)                        │
│   = size_v_row_padded × FLASH_ATTN_BLOCK_SIZE(64) × 2 │
│   예: DV=128 → 동일 32 KB per thread                  │
├───────────────────────────────────────────────────────┤
│ spad_m: Mask 블록 × 2 (옵션)                           │
│   = FLASH_ATTN_BLOCK_SIZE × sizeof(fp16) × 2 (패딩)   │
│   = 256 B per thread                                  │
├───────────────────────────────────────────────────────┤
│ spad_a: VKQ32 누적기                                   │
│   = DV × sizeof(float) (128B 정렬)                     │
│   예: DV=128 → 512 B per thread                       │
└───────────────────────────────────────────────────────┘
```

### 3.3 K/V 시퀀스 길이에 대한 블록 타일링

> 코드 위치: [htp/flash-attn-ops.c:22](ggml/src/ggml-hexagon/htp/flash-attn-ops.c)

```c
#define FLASH_ATTN_BLOCK_SIZE (32 * 2)  // 64행 단위로 K/V를 처리
```

Flash Attention은 **K/V의 시퀀스 축(ne[1])을 64행 블록으로 분할**하여 처리합니다. 이것이 핵심 타일링입니다:

```
K/V 시퀀스 길이 = 2048일 때:
  n_blocks = ceil(2048 / 64) = 32개 블록

각 블록에서:
  1. K 64행을 VTCM에 DMA 로드
  2. V 64행을 VTCM에 DMA 로드
  3. QK^T 계산 (Q 1행 × K 64행)
  4. Online Softmax 누적
  5. P × V 누적
  6. 다음 블록 프리페치 (DMA 파이프라이닝)
```

### 3.4 더블 버퍼링 (DMA 파이프라이닝)

> 코드 위치: [htp/flash-attn-ops.c:379-405](ggml/src/ggml-hexagon/htp/flash-attn-ops.c)

```c
// 처음 2개 블록 프리페치
for (uint32_t ib = 0; ib < MIN(factx->n_blocks, 2); ++ib) {
    dma_queue_push(dma, dma_make_ptr(k_dst, k_src), ...);  // K
    dma_queue_push(dma, dma_make_ptr(v_dst, v_src), ...);  // V
    if (mask) dma_queue_push(dma, ...);                      // Mask
}
```

```
타임라인:
  DMA:    [K₀V₀] [K₁V₁] [K₂V₂] [K₃V₃] ...
  연산:          [QK₀·V₀] [QK₁·V₁] [QK₂·V₂] ...
                ↑ 연산과 DMA가 파이프라인으로 겹침!
```

> 코드 위치: [htp/flash-attn-ops.c:518-535](ggml/src/ggml-hexagon/htp/flash-attn-ops.c)

```c
// 현재 블록 처리 후, ib+2 블록 프리페치
if (ib + 2 < factx->n_blocks) {
    dma_queue_push(dma, dma_make_ptr(k_base, k_src), ...);  // K (ib+2)
    dma_queue_push(dma, dma_make_ptr(v_base, v_src), ...);  // V (ib+2)
    if (mask) dma_queue_push(dma, ...);
}
```

`ib % 2`를 사용하여 **더블 버퍼**를 교대로 사용합니다: 블록 0, 2, 4, ...는 버퍼 A에, 블록 1, 3, 5, ...는 버퍼 B에 로드됩니다.

### 3.5 VTCM 부족 시 에러 처리

> 코드 위치: [htp/flash-attn-ops.c:698-700](ggml/src/ggml-hexagon/htp/flash-attn-ops.c)

```c
if (octx->ctx->vtcm_size < total_spad) {
    return HTP_STATUS_VTCM_TOO_SMALL;
}
```

**Flash Attention에서도 VTCM이 부족하면 에러를 반환합니다.** 현재 스레드 수를 줄이는 등의 자동 조절 메커니즘은 없습니다.

---

## 4. [784, 2048] 입력에 대한 VTCM 초과 시나리오 구체 분석

### 4.1 시나리오: MatMul (Q4_0 가중치 × F32 입력)

가중치: `src0 = [784, N]` (Q4_0), 입력: `src1 = [784, 2048]` (F32)

VTCM 요구량 계산:

```
src1 양자화 후 크기:
  q8x4x2_row_size(784) = hex_round_up(784 + ceil(784/256)*8*2, 128)
                        = hex_round_up(784 + 4*16, 128) = hex_round_up(848, 128) = 896 B
  src1_spad = 896 × 2048 = 1,835,008 B ≈ 1.75 MB  ✓ 이 경우 VTCM에 들어감

src0_spad (6 스레드 기준):
  src0_row_size_padded = hex_round_up(sizeof(block_q4_0) * 784/32, 128) 
                       ≈ hex_round_up(18 * 24.5, 128) ≈ 512 B
  src0_spad = 16 × 512 × 6 = 49,152 B ≈ 48 KB

dst_spad: 2 × dst_row_size × 6 (약간)

총합: ~1.85 MB → 8 MB VTCM에 들어감 ✓
```

**784차원에서는 Q4_0/Q8_0 양자화 경로에서는 실제로 VTCM에 들어갑니다.** 문제가 되는 것은:

### 4.2 시나리오: MatMul (F16 가중치 × F32 입력)

가중치: `src0 = [784, N]` (F16), 입력: `src1 = [784, 2048]` (F32)

```
src1 F16 변환 후 크기:
  f16_src1_row_size = hex_round_up(784 * 2, 128) = 1664 B
  src1_spad = 1664 × 2048 = 3,407,872 B ≈ 3.25 MB

src0_spad: 16 × hex_round_up(784*2, 128) × 6 = 16 × 1664 × 6 = 159,744 B ≈ 156 KB

dst_spad: ~small

f16_total_size = 3.25 MB + 156 KB + ... ≈ 3.4 MB → 8 MB에 들어감 ✓
```

### 4.3 시나리오: src1이 훨씬 큰 경우 — 실제 VTCM 초과

입력이 `[4096, 2048]` (F16-F16 경로)인 경우:

```
src1_spad = hex_round_up(4096*2, 128) × 2048 = 8192 × 2048 = 16,777,216 B = 16 MB > 8 MB ✗
→ matmul_4d() 폴백 (DMA 파이프라이닝 없음, 성능 저하)
```

Q4_0 가중치 × F32 입력 `[4096, 2048]`:
```
src1_spad = q8x4x2_row_size(4096) × 2048 ≈ 4352 × 2048 = 8,912,896 ≈ 8.5 MB > 8 MB ✗
→ HTP_STATUS_VTCM_TOO_SMALL 에러 반환!
```

### 4.4 시나리오: Flash Attention

Q: `[DK, 1]` F16, K: `[DK, seq_len]` F16, V: `[DV, seq_len]` F16  
DK=DV=128, n_threads=6, mask 있음

```
per-thread 요구량:
  spad_q = 256 B
  spad_k = 256 × 64 × 2 = 32,768 B = 32 KB  
  spad_v = 256 × 64 × 2 = 32,768 B = 32 KB
  spad_m = 256 B
  spad_a = 512 B

per-thread 합계 = 65,792 B ≈ 64 KB
6 스레드 합계 = 394,752 B ≈ 386 KB → 8 MB에 여유롭게 들어감 ✓
```

**Flash Attention은 블록 타일링(64행) 덕분에 시퀀스 길이에 무관하게 VTCM 요구량이 일정합니다!** `seq_len=2048`이든 `seq_len=128000`이든 동일한 VTCM만 사용합니다.

단, **DK/DV가 매우 클 때** (예: DK=784) 문제가 될 수 있습니다:

```
DK=784, DV=784, n_threads=6:
  spad_q = hex_round_up(784*2, 128) = 1664 B
  spad_k = hex_round_up(784*2, 128) × 64 × 2 = 1664 × 128 = 212,992 B ≈ 208 KB
  spad_v = 동일 ≈ 208 KB
  spad_m = 256 B
  spad_a = hex_round_up(784*4, 128) = 3200 B

per-thread ≈ 417 KB
6 스레드 ≈ 2.44 MB → 8 MB에 들어감 ✓
```

**DK=4096일 때**:
```
spad_k = hex_round_up(4096*2, 128) × 64 × 2 = 8192 × 128 = 1,048,576 = 1 MB per thread
6 스레드 → K만 6 MB → VTCM 초과 ✗
```

---

## 5. 현재 구현에서 이미 적용된 타일링 기법 정리

| 기법 | 적용 대상 | 핵심 상수 | 설명 |
|------|---------|---------|------|
| **src0 행 스트리밍** | MatMul 2d/matvec | `MM_SPAD_SRC0_NROWS=16` | src0를 16행씩 VTCM으로 DMA → 연산 → 다음 16행 프리페치 |
| **K/V 블록 타일링** | Flash Attention | `FLASH_ATTN_BLOCK_SIZE=64` | K/V를 64행 블록으로 분할, 더블 버퍼링으로 DMA 파이프라이닝 |
| **2×2 벡터 타일링** | MatMul 내적 | `vec_dot_2x2` | src0 2행 × src1 2열을 동시 계산 (데이터 재사용) |
| **동적 행 수 조절** | Binary/Activation Ops | `vtcm_size / (n_threads * row_size)` | VTCM 크기에 맞춰 한 번에 처리할 행 수 자동 조절 |
| **blck 타일링** | matmul_4d | `blck_0=64, blck_1=64` | 루프 블로킹으로 캐시 재사용 향상 (VTCM 미사용) |

---

## 6. VTCM에 맞지 않는 경우를 위한 src1 분할 전략 제안

### 6.1 현재 상태의 한계

현재 양자화 경로(`Q4_0/Q8_0/MXFP4`)에서 src1이 VTCM에 안 들어가면 **에러를 반환**합니다. F16 경로만 `matmul_4d()` 폴백이 있지만 DMA 파이프라이닝이 없어 성능이 크게 떨어집니다.

### 6.2 제안: src1 행 분할 (Column-Block Tiling)

src1의 행(열) 축을 여러 청크로 나누어 순차 처리하는 방식:

```
src1 = [K, M]  (K=특성 차원, M=토큰 수)
M이 너무 크면:

chunk_size = (VTCM_available) / q8x4x2_row_size(K)
n_chunks = ceil(M / chunk_size)

for each chunk c:
  1. src1[c*chunk_size : (c+1)*chunk_size] 양자화 → VTCM
  2. src0 전체 × src1_chunk 행렬곱 수행 (matmul_2d 로직)
  3. 결과를 dst[c*chunk_size : (c+1)*chunk_size] 행에 직접 기록
```

**장점**: 기존 `matmul_2d()` 내부 로직을 그대로 재사용 가능
**주의**: src0는 양자화된 가중치이므로 청크 간 DMA 재로드 필요 없음 (DRAM 직접 스트리밍)

### 6.3 제안: Flash Attention의 VTCM 적응적 스레드 수 조절

DK가 매우 큰 경우, 스레드 수를 줄여 per-thread VTCM 사용량을 감소시킬 수 있습니다:

```c
// 현재 코드 (flash-attn-ops.c:698)
if (octx->ctx->vtcm_size < total_spad) {
    return HTP_STATUS_VTCM_TOO_SMALL;
}

// 개선 제안:
while (octx->ctx->vtcm_size < total_spad && n_threads > 1) {
    n_threads--;
    // spad 크기 재계산
    total_spad = recalculate_spad(n_threads);
}
if (octx->ctx->vtcm_size < total_spad) {
    return HTP_STATUS_VTCM_TOO_SMALL;  // 1 스레드로도 안 되면 에러
}
```

### 6.4 제안: FLASH_ATTN_BLOCK_SIZE 동적 조절

DK가 클 때 블록 크기를 줄여 per-block VTCM 사용량을 감소:

```c
// 현재: 고정 64
#define FLASH_ATTN_BLOCK_SIZE (32 * 2)  // 64

// 개선: 동적 계산
uint32_t block_size = FLASH_ATTN_BLOCK_SIZE;
size_t per_thread_budget = octx->ctx->vtcm_size / n_threads;
while (block_size > 32) {
    size_t needed = calculate_spad_per_thread(block_size, DK, DV);
    if (needed <= per_thread_budget) break;
    block_size /= 2;  // 64 → 32로 축소
}
```

---

## 7. 결론 및 요약

### 현재 구현 상태

1. **MatMul**: src0는 16행 DMA 스트리밍으로 타일링됨. **src1은 전체 VTCM 로딩 필요** → 크기 초과 시 Q4/Q8/MXFP4 경로는 에러 반환, F16만 성능 저하 폴백(`matmul_4d`) 가능
2. **Flash Attention**: K/V를 64행 블록으로 분할 + 더블 버퍼 DMA → **시퀀스 길이에 무관하게 일정한 VTCM 사용** → DK/DV가 극단적으로 클 때만 VTCM 초과 가능
3. **Binary/Activation Ops**: VTCM 크기에 따라 동적으로 처리 행 수 조절 → 타일링이 자동 적용됨

### [784, 2048] 입력 기준

| 경로 | VTCM 요구량 | 8MB VTCM에 적합? |
|------|------------|------------------|
| Q4_0 × F32 MatMul | ~1.85 MB | **적합** ✓ |
| F16 × F32 MatMul | ~3.4 MB | **적합** ✓ |
| F16 × F16 MatMul | ~3.4 MB | **적합** ✓ |
| Flash Attention (DK=784) | ~2.44 MB (6스레드) | **적합** ✓ |

### 실제 VTCM 초과가 발생하는 입력 크기

| 경로 | VTCM 초과 시작 | 비고 |
|------|----------------|------|
| Q4_0/Q8_0 MatMul | `ne[0] ≈ 4096` 이상, `src1_nrows ≥ 2048` | q8x4x2 양자화 후 ≈ 8.5 MB |
| F16 MatMul | `ne[0] × src1_nrows × 2 > ~7 MB` | `matmul_4d` 로 자동 폴백 |
| Flash Attention | `DK ≥ ~2048` (6스레드 기준) | 에러 반환, 스레드 수 줄이면 가능 |

### Flash Attention 사용 시 필수 설정

```bash
export GGML_HEXAGON_EXPERIMENTAL=1
```

이 설정 없이는 `supports_op()`이 Flash Attention을 거부하여 CPU 폴백됩니다.
