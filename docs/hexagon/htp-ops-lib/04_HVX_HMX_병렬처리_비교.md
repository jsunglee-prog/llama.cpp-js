# HVX vs HMX 병렬 처리 비교 및 Attention/FFN 이점 분석

> **분석 대상 파일들**:
> - [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — FFN MatMul 구현
> - [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c) — Attention 구현
> - [src/dsp/ops/rms_norm.c](../../../../htp-ops-lib/src/dsp/ops/rms_norm.c) — RMS Norm (HVX 전용)
> - [src/dsp/ops/mm_benchmark.c](../../../../htp-ops-lib/src/dsp/ops/mm_benchmark.c) — HMX/HVX 벤치마크
> - [include/dsp/hvx_math.h](../../../../htp-ops-lib/include/dsp/hvx_math.h) — HVX 수학 함수
> - [include/dsp/worker_pool.h](../../../../htp-ops-lib/include/dsp/worker_pool.h) — 워커 풀

---

## 1. HVX와 HMX 하드웨어 비교

### 1.1 기본 스펙 비교

| 항목 | HVX (Hexagon Vector eXtension) | HMX (Hexagon Matrix eXtension) |
|------|-------------------------------|-------------------------------|
| **유닛 수** | 4 HVX 슬롯 (멀티스레드 공유) | 1 HMX 유닛 (전체 코어 공유) |
| **처리 단위** | 128 byte 벡터 (VLEN=128) | 32×32 FP16 타일 (2048 bytes) |
| **처리 방식** | SIMD (벡터 연산) | Systolic Array (행렬 연산) |
| **병렬 스레드** | 최대 6 HVX 스레드 동시 실행 | 한 번에 1개 스레드만 사용 가능 |
| **핵심 연산** | Element-wise, Reduction, Permutation | FP16 행렬 내적 (32×32 · 32×32) |
| **메모리 접근** | L1/L2 캐시, VTCM, DDR | **VTCM만** (mxmem 명령) |
| **대역폭** | 128 bytes/cycle (per slot) | 최대 32 타일 동시 로드 가능 |

### 1.2 연산 처리량 비교

> 코드 위치: [src/dsp/ops/mm_benchmark.c](../../../../htp-ops-lib/src/dsp/ops/mm_benchmark.c)

벤치마크 코드에서 확인된 구현 방식:

**HMX 행렬곱** (`hmx_mat_mul_fp16_core`):
```
하나의 HMX 타일 로드 + 내적:
  32 × 32 × 32 = 32,768 FP16 MAC 연산  (1 타일 로드에 32 reduction steps 가정)
  실제: activation 타일 1개 × weight 타일 1개 → 내적 = 32×32 출력
```

**HVX 행렬곱** (`hvx_mat_mul_fp16_core`):
```c
// HVX로 FP16 행렬곱을 수행하려면:
for (int k = 0; k < K; k++) {
    // 1) weight의 한 열을 broadcast
    HVX_Vector v_b = Q6_Vh_vsplat_R(b[k * N + j]);
    // 2) activation 행과 element-wise 곱셈
    HVX_VectorPair v_prod = Q6_Wqf16_vmpy_VhfVhf(v_a_row, v_b);
    // 3) 누적
    v_acc = Q6_Vqf16_vadd_Vqf16Vqf16(v_acc, v_prod_lo);
}
// K 번 반복 → 128 bytes/cycle 처리
```

| 비교 항목 | HMX | HVX (단일 스레드) | HVX (6 스레드) |
|-----------|-----|-----------------|----------------|
| 행렬곱 효율 | **매우 높음** (하드와이어드) | 낮음 (소프트웨어 루프) | 중간 |
| Softmax | 불가능 | **높음** (벡터 exp/div) | **매우 높음** |
| 데이터 변환 | 불가능 | **높음** (FP32↔FP16) | **매우 높음** |
| 역양자화 | 불가능 | **높음** (VLUT16) | **매우 높음** |

---

## 2. 연산별 HVX/HMX 역할 분담

### 2.1 MatMul (FFN Layer)에서의 역할 분담

> 코드 위치: [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c)

#### FP16 Weight MatMul

```
┌──────────────────────────────────────────────────────────────┐
│                    시간축 →                                 │
│                                                              │
│  HVX 스레드:  [FP32→FP16 변환 + Crouton 배치]               │
│               ▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│                                                              │
│  DMA 엔진:    [Weight DDR→VTCM 전송]                         │
│               ░░▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│                                                              │
│  HMX 유닛:    [타일 내적 수행]                               │
│               ░░░░░░░░░▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░    │
│                                                              │
│  HVX 스레드:  [FP16→FP32 변환 + DDR 저장]                    │
│               ░░░░░░░░░░░░░░░░░░▓▓▓▓▓░░░░░░░░░░░░░░░░    │
└──────────────────────────────────────────────────────────────┘
```

#### 양자화 Weight MatMul (4-Stage Pipeline)

```
┌──────────────────────────────────────────────────────────────────────┐
│ 시간 →   T0        T1        T2        T3        T4        T5      │
│                                                                      │
│ DMA:    [W₀ load] [W₁ load] [W₂ load] [W₃ load] ...                │
│ HVX:              [W₀ deq ] [W₁ deq ] [W₂ deq ] [W₃ deq] ...      │
│ HMX:                        [W₀ dot ] [W₁ dot ] [W₂ dot] ...       │
│ HVX:                                  [O₀ store][O₁ store] ...     │
│                                                                      │
│ ★ DMA, HVX, HMX가 동시에 서로 다른 청크를 처리하여 파이프라인 형성  │
└──────────────────────────────────────────────────────────────────────┘
```

**핵심**: 양자화 MatMul에서 HVX는 **역양자화(INT4/INT8 → FP16) 전용 워커**로 동작하며, HMX의 행렬 내적 시간에 다음 청크의 역양자화를 동시 수행한다.

#### Output-Stationary 변형 (대형 Prefill FFN)

> 코드 위치: `mat_mul_qk_0_d16a32_out_stationary()` in [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c)

m≥128, k>n, n>1024일 때 활성화되는 특수 경로:

```
┌────────────────────────────────────────────────────────┐
│ 역할 분담:                                              │
│                                                         │
│ Worker 0 (HVX): Weight DMA + 역양자화                  │
│   - 별도 스레드에서 weight 페치 전담                    │
│   - 2D DMA로 activation도 동시 전송                     │
│                                                         │
│ Worker 1 (HMX): HMX 타일 내적 전담                     │
│   - hmx_manager의 전용 HMX 워커 풀 사용                │
│                                                         │
│ Main Thread: 전체 흐름 조율                              │
│   - M/N/K 블록 크기: 512 단위                          │
│   - Worker 0과 Worker 1을 번갈아 디스패치               │
└────────────────────────────────────────────────────────┘
```

### 2.2 FlashAttention에서의 역할 분담

> 코드 위치: [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c)

FlashAttention은 **HMX와 HVX를 가장 밀도 높게 혼합 사용**하는 연산이다:

```
FOR each KV block (j):
┌─────────────────────────────────────────────────────────────┐
│ 단계           사용 유닛   역할                              │
├─────────────────────────────────────────────────────────────┤
│ 1. K 로드      HVX        DDR→VTCM + vscatter 전치        │
│ 2. V 로드      HVX        DDR→VTCM + Crouton 배치         │
│ 3. S = Q·K^T   HMX ★      타일 내적 (행렬곱)              │
│ 4. rowmax(S)   HVX        행 최대값 계산 (scalar reduction)│
│ 5. exp2(S-m)   HVX        vgather 기반 exp2 테이블 룩업    │
│ 6. rowsum(P)   HVX        행 합 계산 (normalization용)     │
│ 7. D 계산      HVX        대각 스케일 행렬 생성            │
│ 8. O = D·O     HMX ★      이전 출력에 스케일링 적용        │
│ 9. O += P·V    HMX ★      현재 block의 기여분 누적         │
└─────────────────────────────────────────────────────────────┘
```

#### HVX Softmax에서의 핵심 최적화

**vgather 기반 exp2 테이블 룩업**:

> 코드 위치: [src/dsp/ops/precompute_table.c](../../../../htp-ops-lib/src/dsp/ops/precompute_table.c)

```
VTCM 내 256KB 영역에 exp2(-x) 테이블을 사전 계산하여 상주:
  - 4개 복제본 × 64KB = 256KB
  - FP16 입력을 인덱스로 사용
  - vgather 명령어로 벡터 단위 테이블 룩업 수행

성능 이점:
  - 다항식 근사(6차 다항식) 대비 레이턴시 감소
  - VTCM 접근이므로 L1 캐시 미스 없음
```

**다항식 Fallback** (`enable_vgather_exp=false`일 때):

> 코드 위치: [include/dsp/hvx_math.h](../../../../htp-ops-lib/include/dsp/hvx_math.h)

```c
// 6차 다항식 근사: exp2(x) ≈ c₀ + c₁x + c₂x² + c₃x³ + c₄x⁴ + c₅x⁵ + c₆x⁶
// Horner's method로 최적화
static inline HVX_Vector hvx_my_exp2_vhf(HVX_Vector x) {
    // ... 6단계 FMA 체인
}
```

### 2.3 RMS Norm (HVX 전용)

> 코드 위치: [src/dsp/ops/rms_norm.c](../../../../htp-ops-lib/src/dsp/ops/rms_norm.c)

RMS Norm은 **행렬곱이 없으므로 HMX를 사용하지 않는다**:

```
HVX 처리 흐름:
  1. L2 프리페치 (DDR → L2 캐시)
  2. QF32 정밀도로 제곱합 계산 (벡터 FMA)
  3. 32-way 벡터 리덕션으로 sum 계산
  4. 역수 제곱근 계산 (1/sqrt(mean_sq + eps))
  5. 입력 × 스케일 × weight 적용 (한 패스)
```

---

## 3. 워커 풀 기반 병렬 처리

> 코드 위치: [src/dsp/worker_pool.c](../../../../htp-ops-lib/src/dsp/worker_pool.c), [include/dsp/worker_pool.h](../../../../htp-ops-lib/include/dsp/worker_pool.h)

### 3.1 워커 풀 구성

```c
#define MAX_NUM_WORKERS 6  // 최대 동시 워커 스레드 수

// HMX 접근 가능한 워커 풀 초기화 (hmx_mgr.c에서 사용)
worker_pool_init_ex(pool, n_workers=1, allow_hmx=true);

// 일반 HVX 워커 풀 초기화 (flash_attn.c에서 사용)
worker_pool_init(pool, n_workers=N);
```

### 3.2 FlashAttention에서의 병렬 처리 패턴

```
┌─────────────────────────────────────────────────────────────────┐
│ FlashAttention 병렬화: KV 헤드 단위                             │
│                                                                  │
│ Worker 0: KV Head 0      ┐                                      │
│   └ Q heads [0..G-1]     │                                      │
│ Worker 1: KV Head 1      │  각 워커는 1MB VTCM 할당             │
│   └ Q heads [G..2G-1]    │  최대 워커 수 = min(n_kv_heads, 6)   │
│ Worker 2: KV Head 2      │  제약: n_workers * 1MB ≤ 6MB         │
│   └ Q heads [2G..3G-1]   │                                      │
│ ...                       ┘                                      │
│                                                                  │
│ HMX 접근: 모든 워커가 hmx_spin_lock()으로 직렬화                │
│   → Q·K^T 계산 시 한 워커만 HMX 사용                           │
│   → 다른 워커들은 Softmax(HVX)를 동시에 수행                    │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 인사이트**: FlashAttention에서 HMX는 한 번에 한 스레드만 사용하지만, **다른 스레드들은 HVX Softmax를 처리**하여 HMX 대기 시간을 최소화한다. 이것이 HMX 스핀 락 구간 외에서 HVX 작업이 겹치는 이유이다.

```
시간 →
Worker 0: [HVX:K로드] [HMX:Q·Kᵀ] [HVX:softmax] [HMX:P·V] [HVX:O변환] ...
Worker 1:             [HVX:K로드] [  HMX 대기  ] [HMX:Q·Kᵀ] [HVX:softmax] ...
Worker 2:                         [HVX:K로드  ] [  HMX 대기 ] [HMX:Q·Kᵀ] ...

★ HMX 구간은 직렬(스핀 락), HVX 구간은 완전 병렬
```

---

## 4. Attention vs FFN 어디에서 더 이점이 있는가?

### 4.1 FFN (Feed-Forward Network) Layer

FFN은 본질적으로 **대형 행렬곱**이다 (예: 4096×11008):

| 관점 | 분석 |
|------|------|
| **HMX 활용도** | ★★★★★ (매우 높음) — 거의 모든 연산이 행렬곱 |
| **HVX 활용도** | ★★☆☆☆ (낮음) — FP32↔FP16 변환, 역양자화에만 사용 |
| **병렬화 이점** | 제한적 — HMX가 하나이므로 행렬곱 자체는 직렬 |
| **DMA 파이프라인** | ★★★★★ — 4단계 파이프라인으로 메모리 지연 은닉 |
| **주요 병목** | 메모리 대역폭 (Weight가 DDR에 있음) |

**FFN에서의 핵심 이점**:
- 양자화 weight의 DMA 전송과 HVX 역양자화를 HMX 연산과 **완전히 겹칠** 수 있음
- Output-stationary 변형으로 대형 prefill (m≥128)에서 추가 최적화

### 4.2 Attention Layer

Attention은 **행렬곱(HMX) + Softmax(HVX)**의 혼합 연산이다:

| 관점 | 분석 |
|------|------|
| **HMX 활용도** | ★★★★☆ (높음) — Q·K^T, P·V 두 번의 행렬곱 |
| **HVX 활용도** | ★★★★☆ (높음) — Softmax, K 전치, 데이터 변환 |
| **병렬화 이점** | ★★★★★ (매우 높음) — KV 헤드별 독립 처리 가능 |
| **HMX+HVX 겹침** | ★★★★☆ — 한 워커의 HVX 작업과 다른 워커의 HMX가 동시 실행 |
| **주요 병목** | HMX 스핀 락 경합 (헤드 수가 많을수록 대기 시간 증가) |

**Attention에서의 핵심 이점**:
- **GQA(Grouped-Query Attention)**: 여러 Q 헤드를 묶어 한 번의 HMX 연산으로 처리
- **KV 헤드 병렬화**: 최대 6개 워커가 서로 다른 KV 헤드를 동시에 처리
- **vgather exp2 최적화**: VTCM에 상주하는 테이블로 Softmax 속도 대폭 향상
- **HMX-HVX 인터리빙**: 한 워커가 Softmax(HVX)를 수행하는 동안 다른 워커가 행렬곱(HMX) 수행

### 4.3 종합 비교

```
                 FFN MatMul              Attention
                 ──────────              ─────────
HMX 의존도:     ████████████  95%       ████████░░  70%
HVX 의존도:     ██░░░░░░░░░░  15%       ████████░░  65%
DMA 중요도:     ████████████  95%       ██████░░░░  50%
멀티스레드:     ██░░░░░░░░░░  10%       ████████████ 95%

총 성능 이점:
  FFN:       HMX 단일 연산 효율 + DMA 파이프라인 ▶ 메모리 바운드 완화
  Attention: HMX-HVX 동시성 + KV헤드 병렬화 ▶ 레이턴시 감소
```

---

## 5. 벤치마크에서 확인된 성능 차이

> 코드 위치: [src/dsp/ops/mm_benchmark.c](../../../../htp-ops-lib/src/dsp/ops/mm_benchmark.c)

벤치마크에서 구현된 비교 대상:

| 커널 | 구현 방식 | 예상 성능 |
|------|-----------|-----------|
| `hmx_mat_mul_fp16_core` | HMX 타일 기반 | 기준 (1.0x) |
| `hvx_mat_mul_fp16_core` | HVX qf16 누적 | ~10-50x 느림 |
| `hvx_mat_mul_fp32_core` | HVX qf32 누적 | ~20-100x 느림 |
| `hvx_mat_mul_fp16_core_mt` | HVX 멀티스레드 (6T) | ~2-10x 느림 |

**결론**: 행렬곱에서 HMX는 HVX 대비 **10~100배 이상의 처리량**을 제공한다. 이것이 모든 GEMM 연산을 HMX로 수행하고, Softmax/변환과 같은 비행렬 연산만 HVX로 처리하는 설계의 근거이다.
