# 4. 가중치(Weights) 배치 및 메모리 관리

> 분석 대상 코드: [ggml-hexagon.cpp:220-400](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp) (버퍼 컨텍스트), [ggml-hexagon.cpp:600-1530](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp) (repack 함수)

---

## 4.1 메모리 할당: RPC 메모리 + FastRPC mmap

### 버퍼 할당 흐름

가중치가 Hexagon 버퍼에 로딩될 때의 메모리 할당 과정:

```
ggml_backend_hexagon_buffer_alloc(buft, size)
  └── new ggml_backend_hexagon_buffer_context(sess, size, is_repack)
        │
        ├── 1. rpcmem_alloc2(flags, RPCMEM_DEFAULT_HEAP, size)
        │     → ION/DMA-BUF 기반 공유 메모리 할당
        │     → CPU와 DSP 모두 접근 가능한 물리 메모리
        │
        ├── 2. rpcmem_to_fd(data)
        │     → 파일 디스크립터(fd) 획득 (FastRPC mmap용)
        │
        └── 3. mmap_to(sess)
              └── fastrpc_mmap2(domain_id, fd, data, 0, size, FASTRPC_MAP_FD)
                    → DSP 가상 주소 공간에 매핑
                    → DSP에서도 같은 물리 메모리에 접근 가능
```

### 메모리 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                    물리 메모리 (DRAM)                          │
│  ┌──────────────────┐                                         │
│  │  ION/DMA-BUF     │ ← rpcmem_alloc2()로 할당               │
│  │  Coherent Buffer │                                         │
│  └────┬────────┬────┘                                         │
│       │        │                                              │
│  ┌────┴───┐ ┌──┴──────┐                                      │
│  │ CPU    │ │ DSP     │ ← fastrpc_mmap2()로 DSP에 매핑       │
│  │ 가상   │ │ 가상    │                                       │
│  │ 주소   │ │ 주소    │                                       │
│  └────────┘ └─────────┘                                       │
└──────────────────────────────────────────────────────────────┘
```

### 두 가지 버퍼 타입

| 속성 | Regular Buffer | Repack Buffer |
|------|---------------|---------------|
| `is_host` | `opt_hostbuf` (기본 true) | **false** |
| 용도 | 입력/출력/임시 텐서 | 모델 가중치 (양자화된 텐서) |
| 데이터 변환 | 없음 | **repack 수행** |
| 스케줄러 동작 | 호스트 버퍼로 인식→CPU도 접근 | 비호스트→DSP 전용으로 인식 |

**Repack Buffer의 핵심 역할**: 모델 가중치를 HVX 최적화 레이아웃으로 변환하여 DSP 전용 메모리에 저장합니다.

---

## 4.2 Repack: HVX 최적화 텐서 레이아웃

### Repack이 필요한 이유

ggml의 기본 양자화 포맷(Q4_0, Q8_0)은 블록 크기 32를 사용합니다:
```
Q4_0 블록 (기본): [scale(f16), quants(16 bytes)] × 32 elements
```

HVX 벡터 폭은 128바이트(1024비트)이므로, 32개 원소 블록은 너무 작아서 HVX 레지스터를 충분히 활용하지 못합니다. **Repack은 8개 블록(256원소)을 묶어 HVX에 최적화된 레이아웃으로 변환합니다.**

### Repack 포맷 구조

#### Q4x4x2 (Q4_0 → 256원소 슈퍼블록)

```
원본 Q4_0 (블록크기 32):
  [scale₀][q₀₋₁₅ ][q₁₆₋₃₁]  [scale₁][q₃₂₋₄₇][q₄₈₋₆₃]  ...  (8블록)

Repack Q4x4x2 (블록크기 256):
  [q₀₋₁₅, q₃₂₋₄₇, q₆₄₋₇₉, q₉₆₋₁₁₁, ...]   ← 양자값(quants)을 먼저
  [..., scale₀, scale₁, scale₂, ...]             ← 스케일값을 뒤에
```

핵심 변환 원리 (위치: [ggml-hexagon.cpp](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)):

```
repack_q4_0_q4x4x2():
  - 행(row) 내의 모든 Q4_0 블록을 순회
  - 양자화값(nibbles)을 먼저 연속 배치: quants-first layout
  - 스케일값(f16)을 뒤에 연속 배치: scales-after layout
  - 블록 크기: QK_Q4_0x4x2 = 256 (8개 원본 블록 = 4×2 패킹)
```

이 "quants-first, scales-after" 레이아웃은 HVX `Q6_Vw_vrmpy_VbVb` (reduce-multiply) 인스트럭션에 최적화되어 있습니다. 연속된 양자값을 한 번에 128바이트 벡터로 로드하여 8개 HVX 벡터에 대한 reduce-multiply를 수행할 수 있습니다.

#### Q8x4x2 (Q8_0 → 256원소 슈퍼블록)

입력 텐서(활성화값)에 대해 동일한 원리로 repack됩니다. src1(입력)은 실행 시점에 DSP에서 동적으로 양자화 + repack됩니다.

#### MXFP4x4x2 (MXFP4 → 256원소 슈퍼블록)

MX-FP4 포맷에 대한 repack. 스케일은 e8m0 형식이며, HVX에서 "좌측 시프트로 FP32 변환" 트릭을 사용합니다:

```c
// matmul-ops.c 내부
r0_d = Q6_V_vdelta_VV(r0_d, expand);    // uint8 → uint32 확장
r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);   // 0xFF 마스크
r0_d = Q6_Vw_vasl_VwR(r0_d, 23);        // 23비트 좌측 시프트 → FP32 지수
```

---

## 4.3 가중치 로딩 파이프라인

```
모델 파일 로딩 (llama_model_load)
  │
  ├── 일반 텐서: regular buffer에 할당
  │     → rpcmem_alloc2() → fastrpc_mmap2()
  │     → CPU/DSP 모두 접근 가능
  │
  └── 양자화 가중치 (Q4_0, Q8_0, MXFP4): repack buffer에 할당
        │
        ├── 1. rpcmem_alloc2() → 공유 메모리 할당
        │
        ├── 2. ggml_backend_hexagon_buffer_set_tensor()
        │     → memcpy()로 기본 데이터 로딩
        │
        └── 3. repack_tensor_to_hexagon()
              ├── Q4_0: repack_q4_0_q4x4x2()  → quants-first layout
              ├── Q8_0: repack_q8_0_q8x4x2()  → quants-first layout
              └── MXFP4: repack_mxfp4_mxfp4x4x2() → quants-first layout
              
              변환 후 데이터는 같은 공유 메모리에 in-place 저장
              → DSP가 바로 HVX 최적화 레이아웃으로 접근 가능
```

### 중요 포인트

1. **제로 카피(Zero-Copy)**: FastRPC mmap을 사용하므로 가중치를 DSP로 별도 복사할 필요 없음. CPU가 repack한 데이터를 DSP가 직접 읽음.

2. **is_host 플래그의 의미**:
   - Repack 버퍼는 `is_host=false`로 선언됨
   - 이는 스케줄러에게 "이 버퍼의 데이터는 Hexagon에서만 사용 가능"이라고 알려줌
   - 따라서 스케줄러는 해당 가중치를 사용하는 Op을 무조건 Hexagon에 할당

3. **Extra Buffer Type**: `ggml_backend_dev_get_extra_bufts` proc address를 통해 repack 버퍼 타입을 등록. 스케줄러가 가중치 텐서에 대해 repack 버퍼 타입을 자동으로 선택하게 함.

---

## 4.4 동적 양자화 (런타임 src1 변환)

가중치(src0)는 로딩 시 한 번 repack되지만, 입력 텐서(src1)는 매 추론마다 달라지므로 DSP에서 실시간으로 양자화합니다:

```
matmul 실행 시 (htp/matmul-ops.c):
  │
  ├── src0 (가중치): 이미 q4x4x2/q8x4x2 형태로 repack됨 → DRAM에서 직접 사용
  │
  └── src1 (입력):
        ├── F32 형태로 DRAM에 존재
        ├── quantize_f32_q8x4x2() 호출 → VTCM에 q8x4x2 형태로 변환 저장
        │    ├── 워커 스레드 풀에서 병렬 양자화
        │    ├── 각 행을 q8x4x2 슈퍼블록으로 변환
        │    └── 결과를 VTCM src1 스크래치패드에 저장
        │
        └── SKIP_QUANTIZE 최적화:
              연속된 MUL_MAT Op이 같은 src1을 사용하면
              양자화를 건너뛰고 VTCM의 기존 데이터 재사용
```
