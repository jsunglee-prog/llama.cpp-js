# HMX Repack(Crouton) 방식 및 크기 분석

> **분석 대상 파일들**:
> - [include/dsp/hmx_utils.h](../../../../htp-ops-lib/include/dsp/hmx_utils.h) — HMX 타일 상수 정의
> - [include/dsp/quants.h](../../../../htp-ops-lib/include/dsp/quants.h) — 양자화 블록 구조체
> - [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — Activation Crouton 재배치, Weight 역양자화
> - [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c) — K 전치(vscatter), Q/V Crouton 배치
> - [src/host/test.c](../../../../htp-ops-lib/src/host/test.c) — Crouton 인덱싱 예시

---

## 1. Crouton Layout이란?

**Crouton**은 Qualcomm HMX가 요구하는 고유한 타일 메모리 배치(data layout)이다. 일반적인 행-우선(row-major) 또는 열-우선(column-major) 배치와 달리, **32×32 타일 단위로 2행씩 인터리브**된 구조를 가진다.

### 1.1 일반 행렬 vs Crouton 배치 비교

**일반 Row-Major 배치** (32×32 FP16 행렬):
```
메모리 주소 순서: [r₀c₀, r₀c₁, ..., r₀c₃₁, r₁c₀, r₁c₁, ..., r₃₁c₃₁]
연속 메모리에 행 단위로 저장
```

**Crouton 배치** (32×32 FP16 타일):
```
메모리 주소 순서: [r₀c₀, r₁c₀, r₀c₁, r₁c₁, ..., r₀c₃₁, r₁c₃₁,    ← 행 0, 1
                  r₂c₀, r₃c₀, r₂c₁, r₃c₁, ..., r₂c₃₁, r₃c₃₁,    ← 행 2, 3
                  ...
                  r₃₀c₀, r₃₁c₀, r₃₀c₁, r₃₁c₁, ..., r₃₀c₃₁, r₃₁c₃₁] ← 행 30, 31
```

**인덱싱 공식** (코드에서 발견):
> 코드 위치: [src/host/test.c](../../../../htp-ops-lib/src/host/test.c) — Crouton 인덱스 계산

```c
// tile[row][col] 접근을 위한 Crouton 인덱스:
int idx = (row & ~1) * 32 + col * 2 + (row & 1);
//        ^^^^^^^^^^^^^^^^ ^^^^^^^^   ^^^^^^^^^^
//        2행 단위 오프셋    열×2      홀짝 행 오프셋
```

| 요소 | 의미 |
|------|------|
| `(row & ~1) * 32` | 2행 묶음의 시작 위치 (행 0,1 → 0, 행 2,3 → 64, ...) |
| `col * 2` | 각 열에 2개 요소(짝수행+홀수행)가 연속 배치 |
| `(row & 1)` | 동일 열 내에서 짝수행=0, 홀수행=1 오프셋 |

### 1.2 시각적 표현

```
32×32 FP16 Crouton Tile (2048 bytes):

메모리 오프셋:  0    2    4    6    8   10   ...  126
             ┌────┬────┬────┬────┬────┬────┬────┬────┐
    행 0+1:  │r0c0│r1c0│r0c1│r1c1│r0c2│r1c2│... │r1c31│  64 bytes
             ├────┼────┼────┼────┼────┼────┼────┼────┤
    행 2+3:  │r2c0│r3c0│r2c1│r3c1│r2c2│r3c2│... │r3c31│  64 bytes
             ├────┼────┼────┼────┼────┼────┼────┼────┤
    ...      │    │    │    │    │    │    │    │    │
             ├────┼────┼────┼────┼────┼────┼────┼────┤
    행 30+31:│r30 │r31 │r30 │r31 │r30 │r31 │... │r31  │  64 bytes
             │ c0 │ c0 │ c1 │ c1 │ c2 │ c2 │    │c31  │
             └────┴────┴────┴────┴────┴────┴────┴────┘

총: 16 행쌍 × 64 bytes = 1024 FP16 요소 × 2 bytes = 2048 bytes
```

---

## 2. HMX 타일 크기 정의

> 코드 위치: [include/dsp/hmx_utils.h](../../../../htp-ops-lib/include/dsp/hmx_utils.h)

```c
#define HMX_FP16_TILE_N_ROWS    32      // 타일 행 수
#define HMX_FP16_TILE_N_COLS    32      // 타일 열 수 (= reduction 축)
#define HMX_FP16_TILE_N_ELMS    1024    // 총 요소 수 (32 × 32)
#define HMX_FP16_TILE_SIZE      2048    // 바이트 크기 (1024 × 2 bytes)
```

### 행렬 곱 C = A × B에서의 타일 구성

```
A (M × K) × B (K × N) = C (M × N)

Activation A:                Weight B:                 Output C:
┌──────────────────┐        ┌──────────────────┐      ┌──────────────┐
│ Tile (32×32)     │ K/32   │ Tile (32×32)     │ K/32 │ Tile (32×32) │
│ n_act_tiles개    │ tiles  │ n_wt_tiles개     │ tiles│ n_out_tiles개│
│ = M/32 개        │        │ = N/32 개        │      │ = M/32 × N/32│
└──────────────────┘        └──────────────────┘      └──────────────┘

축적 (Reduction):
  n_act_tiles × n_wt_tiles → n_out_tiles
  K 차원을 32 단위로 순회하며 축적기에 누적
```

---

## 3. Activation의 Crouton 재배치

> 코드 위치: [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — `transfer_activation_chunk_fp32_to_fp16()` 함수

FP32 Activation은 DDR에 일반 row-major로 저장되어 있으므로, VTCM으로 복사하면서 **FP32→FP16 변환**과 **Crouton 재배치**를 동시에 수행해야 한다.

### 3.1 변환 과정 (2행 단위)

```c
static void transfer_activation_chunk_fp32_to_fp16(
    __fp16 * restrict activation_tiles,     // VTCM 내 Crouton 영역
    const float * restrict activation_data, // DDR 내 FP32 데이터
    int n_rows, int n_cols,
    int stride_k                            // DDR에서의 행 간격
) {
    for (int tile_i = 0; tile_i < n_tiles_m; tile_i++) {
        for (int i = 0; i < 32; i += 2) {  // 2행씩 처리
            __fp16 * dst = activation_tiles + tile_offset;
            const float * src_row0 = activation_data + (i+0) * stride_k;
            const float * src_row1 = activation_data + (i+1) * stride_k;

            for (int tile_j = 0; tile_j < n_tiles_k; tile_j++) {
                // HVX를 이용한 FP32 → FP16 변환
                HVX_Vector v_fp32_row0 = *(HVX_Vector *)src_row0;
                HVX_Vector v_fp32_row1 = *(HVX_Vector *)src_row1;
                HVX_Vector v_fp16 = hvx_my_wsf_to_vhf(v_fp32_row0, v_fp32_row1);
                // 결과: [r0c0, r1c0, r0c1, r1c1, ...] = Crouton 인터리브
                *(HVX_Vector *)dst = v_fp16;
            }
        }
    }
}
```

**핵심**: `hvx_my_wsf_to_vhf()` 함수가 2개의 FP32 벡터(각 32개 float = 128 bytes)를 1개의 FP16 벡터(64개 half = 128 bytes)로 변환하면서, 자연스럽게 **2행 인터리브** Crouton 배치를 생성한다.

> 코드 위치: [include/dsp/hvx_convert.h](../../../../htp-ops-lib/include/dsp/hvx_convert.h) — FP32↔FP16 변환 함수

---

## 4. Weight의 Pre-Permutation (사전 재배치)

### 4.1 FP16 Weight

FP16 Weight는 **호스트(CPU) 측에서 미리 Crouton 포맷으로 재배치(pre-permuted)**되어 DDR에 저장된다. DSP에서는 DMA로 그대로 복사하면 된다.

```
호스트 (사전 처리):
  Row-major FP16 weight → Crouton FP16 weight (pre-permuted)
  
DSP (런타임):
  DDR (pre-permuted) ──DMA──▶ VTCM → HMX (추가 변환 불필요)
```

### 4.2 양자화 Weight (Q4_0/Q8_0/IQ4_NL)

양자화 Weight도 **호스트에서 Crouton 순서로 사전 재배치**된다. 하지만 DSP에서는 역양자화 + FP16 변환이 추가로 필요하다.

> 코드 위치: [include/dsp/quants.h](../../../../htp-ops-lib/include/dsp/quants.h)

#### 양자화 블록 구조

```c
// 커스텀 수퍼블록: QK_K = 256 요소 단위
#define QK_K 256

// Q4_0 수퍼블록
typedef struct {
    __fp16 d[QK_K / 32];    // 8개 스케일 (256/32 = 8)
    uint8_t qs[QK_K / 2];   // 128 bytes (4비트 × 256 = 128 bytes)
} my_block_q4_0;
// 크기: 8×2 + 128 = 144 bytes (256 요소 커버)
// BPW(Bits Per Weight): 144/256 × 8 = 4.5 bits

// Q8_0 수퍼블록
typedef struct {
    __fp16 d[QK_K / 32];    // 8개 스케일
    int8_t qs[QK_K];        // 256 bytes (8비트 × 256)
} my_block_q8_0;
// 크기: 8×2 + 256 = 272 bytes
// BPW: 272/256 × 8 = 8.5 bits
```

#### Q4_0 역양자화 과정

> 코드 위치: [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — `dequantize_permuted_weight_q4_0_to_fp16_hvx_task()` 함수

```
DDR (Q4_0 수퍼블록, pre-permuted Crouton 순서)
  │
  │ DMA 전송
  ▼
VTCM (raw quants)
  │
  │ HVX 역양자화:
  │  1. 스케일(d[]) 추출 → 8개 FP16 스케일
  │  2. 양자화값(qs[]) 추출 → 4비트 값 분리 (하위/상위 니블)
  │  3. VLUT16으로 INT4→FP16 변환 (q4_0_to_fp16_lut 테이블 사용)
  │  4. 스케일 × FP16 양자화값
  ▼
VTCM (FP16 Crouton 타일)
  │
  │ mxmem (HMX 타일 로드)
  ▼
HMX Accumulator (내적 수행)
```

#### VLUT16 기반 INT4→FP16 변환

```c
// Q4_0 → FP16 룩업 테이블 (16개 엔트리)
static const __fp16 q4_0_to_fp16_lut[16] = {
    -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
     0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f
};

// IQ4_NL → FP16 룩업 테이블 (비선형 양자화)
static const __fp16 iq4_nl_to_fp16_lut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};
```

HVX의 `Q6_Vw_equals_vlut16(Vu, Vv, Rt)` 명령어로 **4비트 인덱스 → FP16 값** 변환을 벡터 단위로 수행한다.

---

## 5. FlashAttention에서의 Repack 방식

### 5.1 K 행렬 전치 + Crouton (vscatter)

> 코드 위치: [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c)

K 행렬은 원래 `(n_kv_heads, seq_len, head_dim)` 형태로 DDR에 저장되어 있다. S = Q·K^T 연산에서 K는 우항(weight 역할)이므로, **전치(Transpose) + Crouton 재배치**가 필요하다.

```
원본 K (DDR, row-major):          전치된 K (VTCM, Crouton):
   head_dim →                        seq_len →
  ┌──────────────────┐              ┌──────────────────┐
s │ k₀₀ k₀₁ k₀₂ ... │          h   │ k₀₀ k₁₀ k₂₀ ... │
e │ k₁₀ k₁₁ k₁₂ ... │    ──▶   e   │ k₀₁ k₁₁ k₂₁ ... │
q │ k₂₀ k₂₁ k₂₂ ... │    vscat d   │ k₀₂ k₁₂ k₂₂ ... │
  └──────────────────┘          ↓   └──────────────────┘
                                    (32×32 Crouton tiles)
```

`vscatter` 명령어는 HVX 벡터의 각 FP16 요소를 VTCM 내 지정된 오프셋에 분산 저장하여, 한 번의 벡터 연산으로 전치를 수행한다.

### 5.2 V 행렬 Crouton 배치

V 행렬은 P·V 연산에서 우항(weight 역할)이므로, 열-우선(column-major) Crouton 레이아웃으로 VTCM에 배치된다.

### 5.3 Q 행렬 Crouton 배치

Q 행렬은 좌항(activation 역할)이므로, 행-우선(row-major) Crouton 레이아웃으로 배치된다. FP32→FP16 변환이 동시에 수행된다.

---

## 6. Crouton 포맷 크기 요약

| 항목 | 크기 | 계산 |
|------|------|------|
| 1 FP16 타일 | 2048 bytes | 32 × 32 × 2 bytes |
| M×K Activation tiles | `ceil(M/32) × ceil(K/32) × 2048` bytes | 타일 수 × 타일 크기 |
| K×N Weight tiles | `ceil(K/32) × ceil(N/32) × 2048` bytes | |
| M×N Output tiles | `ceil(M/32) × ceil(N/32) × 2048` bytes | |
| HMX Column Scales | 256 bytes | 32 columns × 4 bytes (packed FP16 pairs) |

### 실제 연산에서의 크기 예시

**MatMul** (LLaMA 7B, m=1, k=4096, n=4096):
| 요소 | 타일 수 | 바이트 |
|------|---------|--------|
| Activation (1×4096) | 1 × 128 = 128 tiles | 256 KB |
| Weight (4096×4096) | 128 × 128 = 16384 tiles | 32 MB (VTCM에 안 들어감 → 타일링 필요) |
| Output (1×4096) | 1 × 128 = 128 tiles | 256 KB |

**FlashAttention** (head_dim=128, Br=64, Bc=256):
| 요소 | 타일 수 | 바이트 |
|------|---------|--------|
| Q tiles (64×128) | 2 × 4 = 8 tiles | 16 KB |
| K tiles (128×256) | 4 × 8 = 32 tiles | 64 KB |
| S tiles (64×256) | 2 × 8 = 16 tiles | 32 KB |
| V tiles (256×128) | 8 × 4 = 32 tiles | 64 KB |
| O tiles (64×128) ×2 | 8 × 2 = 16 tiles | 32 KB |
| 기타 (scales, vectors) | - | ~20 KB |
| **총합** | | **~228 KB** |

---

## 7. 정리: Repack이 필요한 이유

1. **HMX 하드웨어 제약**: `mxmem` 명령어는 Crouton 포맷의 데이터만 처리 가능
2. **2행 인터리브**: HMX 내부의 MAC(Multiply-Accumulate) 유닛이 2행을 동시에 처리하는 구조
3. **VTCM 대역폭 최적화**: 연속 메모리 접근으로 VTCM 읽기 효율 극대화
4. **사전 재배치(Pre-permute)**: Weight는 호스트에서 미리 변환하여 런타임 오버헤드 제거
5. **런타임 재배치**: Activation은 값이 매번 바뀌므로 HVX를 이용해 즉시 변환
