# VTCM 8MB 타일링 전략 분석

> **분석 대상 파일들**:
> - [src/dsp/vtcm_mgr.cc](../../../../htp-ops-lib/src/dsp/vtcm_mgr.cc) — VTCM 관리자 구현
> - [include/dsp/vtcm_mgr.h](../../../../htp-ops-lib/include/dsp/vtcm_mgr.h) — VTCM 관리 API
> - [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — MatMul VTCM 타일링
> - [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c) — FlashAttention VTCM 타일링
> - [src/dsp/ops/precompute_table.c](../../../../htp-ops-lib/src/dsp/ops/precompute_table.c) — exp2 테이블 VTCM 상주

---

## 1. VTCM 메모리 관리 아키텍처

### 1.1 VTCM 전체 할당

> 코드 위치: [src/dsp/vtcm_mgr.cc](../../../../htp-ops-lib/src/dsp/vtcm_mgr.cc)

```c
void vtcm_manager_setup(void) {
    // 전체 VTCM을 1개 페이지로 요청
    compute_res_attr_set_vtcm_param_v2(&attr,
        /*vtcm_size=*/ 0xFFFFFFFF,  // 최대 크기 요청
        /*min_page_size=*/ 0,        // 최소 페이지 제약 없음
        /*min_vtcm_size=*/ 0);       // 최소 크기 제약 없음
    
    HAP_compute_res_acquire(&attr, /*timeout=*/ 100000);
    
    vtcm_base = compute_res_attr_get_vtcm_ptr_v2(&attr);
    vtcm_size = compute_res_attr_get_vtcm_size(&attr);
    // 결과: vtcm_size ≈ 8MB (장치에 따라 다름)
}
```

### 1.2 VTCM 할당 전략: 2가지 방식

```
VTCM 8MB 전체 레이아웃:
┌────────────────────────────────────────────────────────┐ vtcm_base
│                                                         │
│          순차 할당 영역 (Sequential Allocation)         │
│          vtcm_seq_alloc()로 동적 할당                   │
│          (하단에서 상단으로 성장 ↑)                     │
│                                                         │
├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤ 
│                                                         │
│          .... 여유 공간 ....                             │
│                                                         │
├────────────────────────────────────────────────────────┤ top - reserved
│  예약 영역 3: (이름 기반 예약)                          │
│  vtcm_manager_reserve_area()로 예약                     │
│  (상단에서 하단으로 성장 ↓)                             │
├────────────────────────────────────────────────────────┤
│  예약 영역 2                                            │
├────────────────────────────────────────────────────────┤
│  예약 영역 1: "safe_softmax::exp2_hf_qf16" (256KB)     │
└────────────────────────────────────────────────────────┘ vtcm_base + vtcm_size
```

#### 순차 할당 (`vtcm_seq_alloc`)

```c
// 간단한 bump allocator (현재 오프셋에서 정렬 후 할당)
void * vtcm_seq_alloc(vtcm_seq_alloc_t * state, int size, int alignment) {
    int offset = align_up(state->cur_offs, alignment);
    state->cur_offs = offset + size;
    return (char *)state->vtcm_base + offset;
}
```

- **용도**: MatMul, FlashAttention에서 임시 작업 영역 할당
- **특징**: 해제(free) 없음 — 각 연산 호출마다 초기화하고 재사용
- **정렬**: 타일 정렬(2048 bytes) 필요

#### 명명된 예약 영역 (`vtcm_manager_reserve_area`)

```c
void * vtcm_manager_reserve_area(const char * name, int size) {
    // VTCM 꼭대기에서 아래로 할당
    vtcm_reserved_top -= align_up(size, alignment);
    reserved_areas[name] = {vtcm_reserved_top, size};
    return vtcm_reserved_top;
}

void * vtcm_manager_query_area(const char * name) {
    // 이름으로 이미 예약된 영역 조회
    return reserved_areas[name].ptr;
}
```

- **용도**: exp2 테이블 등 프로세스 수명 동안 상주하는 데이터
- **특징**: 한번 예약하면 해제하지 않음, 이름으로 조회 가능

---

## 2. MatMul의 VTCM 타일링 전략

> 코드 위치: [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c)

### 2.1 VTCM 4-영역 분할

```c
#define WEIGHT_AREA_SIZE      (1 * 1024 * 1024)   // 1MB
#define ACTIVATION_AREA_SIZE  (1 * 1024 * 1024)   // 1MB
#define OUTPUT_AREA_SIZE      (1 * 1024 * 1024)   // 1MB
#define SCRATCH_AREA_SIZE     (1 * 1024 * 1024)   // 1MB
// 총 4MB 사용 (나머지 ~4MB는 예약 영역 + 여유)
```

### 2.2 타일 청크 크기 결정 알고리즘

> 함수: `find_chunk_size()` in [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c)

VTCM 영역 크기 제약 하에서 **최대 타일 청크**를 찾는 알고리즘:

```
입력: total_m, total_n, total_k (행렬 차원), area_sizes (VTCM 영역 크기들)
출력: chunk_m, chunk_n, chunk_k (한 번에 처리할 타일 수)

알고리즘:
1. n_act_tiles = min(ceil(total_m/32), min(ACTIVATION_AREA/2048, 32))
   → Activation 타일 수: VTCM과 HMX 제약(max 32) 중 작은 값

2. n_wt_tiles = WEIGHT_AREA / 2048
   → Weight 타일 수: Weight 영역에 맞는 최대 타일 수

3. n_out_tiles = n_act_tiles × n_wt_tiles
   → Output 타일 수: 결과 행렬에 필요한 타일 수

4. IF n_out_tiles × 2048 > OUTPUT_AREA:
     n_wt_tiles를 축소하여 맞춤
   
5. chunk_k = min(WEIGHT_AREA / (n_wt_tiles × 2048), ceil(total_k/32))
   → K 축 청크: Weight 영역 내에서의 reduction 깊이
```

### 2.3 MatMul 타일링 루프 구조

```
전체 행렬곱 C[M×N] = A[M×K] × B[K×N]:

FOR m_offset = 0 to M, step chunk_m:
  FOR n_offset = 0 to N, step chunk_n:
    ┌──────────────────────────────────────────────┐
    │ 1. Activation 청크 전송                       │
    │    DDR[m_offset..m_offset+chunk_m, :]         │
    │    → VTCM ACTIVATION 영역 (FP32→FP16+Crouton)│
    └──────────────────────────────────────────────┘
    
    FOR k_offset = 0 to K, step chunk_k:
      ┌────────────────────────────────────────────┐
      │ 2. Weight 청크 전송                         │
      │    DDR[:, n_offset..n_offset+chunk_n]       │
      │    → VTCM WEIGHT 영역 (DMA)                │
      ├────────────────────────────────────────────┤
      │ 3. HMX 내적                                │
      │    VTCM(Act) × VTCM(Wt) → VTCM(Out)       │
      │    Accumulate over K chunks                │
      └────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────┐
    │ 4. 출력 저장                                  │
    │    VTCM OUTPUT 영역 → DDR[m_offset, n_offset]│
    │    (FP16→FP32 변환)                          │
    └──────────────────────────────────────────────┘
```

### 2.4 양자화 MatMul의 더블 버퍼링

양자화 weight의 경우 **DMA + 역양자화 + HMX**를 겹치기 위해 더블 버퍼를 사용:

```
         Weight 영역 (1MB)               Scratch 영역 (1MB)
       ┌────────┬────────┐           ┌────────┬────────┐
iter 0 │▓ DMA A │        │           │        │        │
iter 1 │        │▓ DMA B │           │▓ Deq A │        │
iter 2 │▓ DMA A │  read  │           │  read  │▓ Deq B │  ← HMX reads A
iter 3 │  read  │▓ DMA B │           │▓ Deq A │  read  │  ← HMX reads B
       └────────┴────────┘           └────────┴────────┘

▓ = 현재 쓰기 중  /  read = HMX 또는 HVX 읽기 중

타일링 제약:
  - 양자화 Weight 청크 1개: DMA 버퍼 크기 ≤ 512KB (반쪽)
  - 역양자화 결과 생크 1개: FP16 타일 크기 ≤ 512KB (반쪽)
```

---

## 3. FlashAttention의 VTCM 타일링 전략

> 코드 위치: [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c)

### 3.1 워커당 VTCM 할당

```c
// 최대 6개 워커, 각 1MB
int n_workers = min(n_kv_heads, 6);
// 제약: n_workers * 1MB ≤ available_vtcm - reserved_areas
int worker_vtcm_size = 1 * 1024 * 1024;  // 1MB per worker
```

**전체 VTCM 사용 구조**:
```
VTCM 8MB:
┌──────────────────────┐ Base
│  Worker 0: 1MB       │ ← KV Head 0
├──────────────────────┤
│  Worker 1: 1MB       │ ← KV Head 1
├──────────────────────┤
│  Worker 2: 1MB       │ ← KV Head 2
├──────────────────────┤
│  Worker 3: 1MB       │ ← KV Head 3
├──────────────────────┤
│  Worker 4: 1MB       │ ← KV Head 4
├──────────────────────┤
│  Worker 5: 1MB       │ ← KV Head 5
├──────────────────────┤ 6MB
│  여유 공간 (~1.75MB) │
├──────────────────────┤ ~7.75MB
│  exp2 테이블 (256KB) │ ← 예약 영역 (항상 상주)
└──────────────────────┘ 8MB
```

### 3.2 동적 Br/Bc 결정 알고리즘

> 함수: `fa_f16_find_chunk_size()` in [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c)

주어진 1MB VTCM 예산 내에서 **Br(행 블록 크기)과 Bc(열 블록 크기)**를 동적으로 결정:

```
알고리즘 (Greedy):

1. 초기값: Br = 32, Bc = 64 (최소)
2. 상수: MAX_G_BR = 256 (GQA 포함 최대 행 블록)

3. PHASE 1 — Br 확장 (우선):
   WHILE vtcm_usage(Br+32, Bc) ≤ available_vtcm
     AND (Br+32) ≤ MAX_G_BR:
       Br += 32

4. PHASE 2 — Bc 확장:
   WHILE vtcm_usage(Br, Bc+64) ≤ available_vtcm:
       Bc += 64

결과: 주어진 VTCM 내에서 가능한 최대 Br, Bc
```

### 3.3 VTCM 사용량 계산 공식

> 함수: `fa_f16_compute_vtcm_usage()` in [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c)

```
VTCM 사용량(bytes) =
  Q tiles:    align(g_br, 32)/32 × D/32 × 2048         // Q 타일
+ O tiles:    align(g_br, 32)/32 × D/32 × 2048 × 2     // O 더블 버퍼
+ K tiles:    Bc/32 × D/32 × 2048                       // K 타일 (전치됨)
+ V tiles:    D/32 × Bc/32 × 2048                       // V 타일
+ S tiles:    align(g_br, 32)/32 × Bc/32 × 2048         // S = Q·K^T
+ P tiles:    align(g_br, 32)/32 × Bc/32 × 2048         // P = softmax(S)
+ D tiles:    align(g_br, 32)/32 × align(g_br, 32)/32 × 2048  // 대각 스케일
+ Col vectors: 4 × align(g_br, 32) × 2                  // m, l, rowmax, rowsum
+ Row buffers: 2 × Bc × 2                               // softmax 작업 버퍼
+ HMX scales:  2 × 256                                  // column scales

여기서:
  g_br = align_up(n_gqa_groups × Br, 32)  // GQA 팩터 반영
  D = head_dim
```

### 3.4 실제 크기 예시

**LLaMA-7B** (head_dim=128, n_gqa=1):

| 파라미터 | 값 | 계산 |
|----------|-----|------|
| Br | 64 | Phase 1에서 결정 |
| Bc | 256 | Phase 2에서 결정 |
| g_br | 64 | align(1×64, 32) = 64 |
| Q tiles | 16 KB | 2 × 4 × 2048 |
| O tiles ×2 | 32 KB | 2 × 4 × 2048 × 2 |
| K tiles | 64 KB | 8 × 4 × 2048 |
| V tiles | 64 KB | 4 × 8 × 2048 |
| S tiles | 32 KB | 2 × 8 × 2048 |
| P tiles | 32 KB | 2 × 8 × 2048 |
| D tiles | 8 KB | 2 × 2 × 2048 |
| Col vectors | 512 B | 4 × 64 × 2 |
| Row buffers | 1 KB | 2 × 256 × 2 |
| Scales | 512 B | 2 × 256 |
| **총합** | **~249 KB** | **1MB 내 충분** |

**GQA 모델** (head_dim=128, n_gqa=8):

| 파라미터 | 값 |
|----------|-----|
| Br | 32 (GQA로 인해 g_br가 커지므로 Br 감소) |
| g_br | 256 (align(8×32, 32) = 256) |
| Q tiles | 128 KB (8 × 4 × 2048) |
| O tiles ×2 | 256 KB |
| S tiles | 128 KB |
| P tiles | 128 KB |
| D tiles | 128 KB |
| **총합** | **~900 KB** (1MB 근접) |

---

## 4. exp2 테이블의 VTCM 상주 전략

> 코드 위치: [src/dsp/ops/precompute_table.c](../../../../htp-ops-lib/src/dsp/ops/precompute_table.c)

### 4.1 테이블 구조

```c
// VTCM 예약: 256KB
const char * AREA_NAME = "safe_softmax::exp2_hf_qf16";
const int AREA_SIZE = 4 * 64 * 1024;  // 4 × 64KB = 256KB

// 4개 복제본이 필요한 이유:
// vgather 명령어는 64KB 범위 내에서만 인덱싱 가능
// FP16 음수 값 범위를 커버하려면 4개 세그먼트 필요
```

### 4.2 계산 방식

```
exp2(x) 테이블 사전 계산:
  - 입력: FP16 음수 값 (-15.0 ~ 0.0 범위)
  - 계산: FP32 정밀도로 exp2(x) 계산 후 FP16으로 변환
  - safe softmax: exp2(s - max(s)) 형태이므로 입력은 항상 ≤ 0
  - VTCM에 영구 상주하여 vgather 명령으로 테이블 룩업
```

### 4.3 성능 이점

| 방식 | 지연시간 | VTCM 사용 |
|------|---------|-----------|
| HVX 다항식 근사 (6차) | ~30+ cycles/vector | 0 |
| vgather 테이블 룩업 | ~5 cycles/vector | 256KB |

256KB VTCM을 투자하여 **6배 이상의 Softmax 속도 향상**을 달성한다.

---

## 5. Output-Stationary MatMul의 VTCM 전략

> 코드 위치: `mat_mul_qk_0_d16a32_out_stationary()` in [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c)

대형 prefill (m≥128, k>n, n>1024)에서 사용되는 특수 타일링:

```
M, N, K 블록 크기: 512 단위

VTCM 분할:
┌───────────────────────────────────────┐
│ Output 타일 (512×512 결과 상주)       │ ← K 루프 동안 VTCM에 유지
│  = (512/32) × (512/32) = 256 타일    │
│  = 256 × 2048 = 512KB               │
├───────────────────────────────────────┤
│ Activation 청크 (2D DMA로 전송)       │
│  512×512 FP16 = 512KB               │
├───────────────────────────────────────┤
│ Weight 청크 (DMA + 역양자화)          │
│  더블 버퍼링                          │
└───────────────────────────────────────┘

핵심 차이: Output을 VTCM에 유지하면서 K 차원을 순회
  → DDR로의 중간 결과 저장/로드 불필요
  → K가 큰 경우 (FFN의 hidden_dim 축소) 특히 유리
```

---

## 6. VTCM 타일링 전략 비교 요약

| 전략 | MatMul (기본) | MatMul (Out-Stationary) | FlashAttention |
|------|:----:|:----:|:----:|
| **총 VTCM 사용** | 4MB (고정 분할) | 가변 (최대 4MB) | n_workers × 1MB |
| **분할 방식** | 4 × 1MB 고정 영역 | M/N/K 블록 크기 기반 | 워커별 독립 1MB |
| **버퍼링** | 더블 버퍼 (양자화) | 더블 버퍼 | 더블 버퍼 (O 타일) |
| **Br/Bc 결정** | find_chunk_size() | 고정 512 | fa_f16_find_chunk_size() |
| **K 축 누적** | DDR에 중간 저장 | VTCM에 유지 (!) | VTCM에 유지 |
| **멀티스레드** | 단일 + HMX 워커 | 2 워커 (fetch + compute) | 최대 6 워커 병렬 |
| **상주 데이터** | - | - | exp2 테이블 (256KB) |

### 핵심 설계 원칙

1. **VTCM 크기 인식(Size-Aware) 타일링**: 런타임에 사용 가능한 VTCM 크기를 확인하고, 그에 맞는 최대 타일 크기를 동적으로 결정
2. **파이프라인 겹침**: DMA 전송, HVX 변환, HMX 연산이 서로 다른 VTCM 영역에서 동시 동작
3. **상주 데이터 최소화**: exp2 테이블(256KB)만 영구 상주, 나머지는 연산별로 재사용
4. **정렬 보장**: 모든 VTCM 할당은 2048 bytes(타일 크기)에 정렬, DMA 디스크립터는 64 bytes에 정렬
