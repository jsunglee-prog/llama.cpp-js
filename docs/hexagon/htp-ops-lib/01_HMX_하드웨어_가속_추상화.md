# HMX 하드웨어 가속 추상화 분석

> **분석 대상 파일들**:
> - [include/dsp/hmx_utils.h](../../../../htp-ops-lib/include/dsp/hmx_utils.h) — HMX 인라인 어셈블리 래퍼
> - [include/dsp/hmx_mgr.h](../../../../htp-ops-lib/include/dsp/hmx_mgr.h) — HMX 자원 관리 API
> - [src/dsp/hmx_mgr.c](../../../../htp-ops-lib/src/dsp/hmx_mgr.c) — HMX 자원 관리 구현
> - [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — HMX를 사용하는 행렬 곱셈
> - [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c) — HMX를 사용하는 FlashAttention

---

## 1. HMX(Hexagon Matrix eXtension)란?

HMX는 Qualcomm Hexagon DSP v73 이상에 내장된 **FP16 행렬 연산 가속 유닛**이다. GPU의 Tensor Core와 유사하게, **타일(Tile) 단위**로 행렬 내적(dot product)을 하드웨어 수준에서 수행한다.

### 핵심 스펙

| 항목 | 값 | 비고 |
|------|-----|------|
| 데이터 타입 | FP16 (IEEE 754 Half) | 입력/출력 모두 FP16 |
| 타일 크기 | 32 × 32 = 1024 요소 | 1 타일 = 2048 바이트 |
| 타일 레이아웃 | **Crouton Layout** | 논문 Figure 4 참조 |
| 축적기(Accumulator) | 타일 면적의 Column 단위 | 최대 32개 타일 동시 로드 가능 |
| 출력 스케일링 | Per-Column Bias (column scale) | 누적 결과에 대한 스케일 적용 |

---

## 2. HMX 추상화 계층 구조

```
┌────────────────────────────────────────────────────────────┐
│                  연산자 (Operator) 레벨                     │
│     mat_mul.c        flash_attn.c        mm_benchmark.c    │
├────────────────────────────────────────────────────────────┤
│              HMX 유틸리티 레벨 (hmx_utils.h)               │
│   hmx_load_tiles_fp16()    hmx_dot_fp16()                  │
│   hmx_consume_accumulator_fp16()   hmx_set_output_scales() │
├────────────────────────────────────────────────────────────┤
│              HMX 자원 관리 레벨 (hmx_mgr.h/c)             │
│   hmx_manager_setup()  hmx_manager_enable_for_thread()     │
│   hmx_manager_get_worker_pool()  hmx_spin_lock()           │
├────────────────────────────────────────────────────────────┤
│            Qualcomm HAP API (OS 수준)                      │
│   HAP_compute_res_acquire()  HAP_compute_res_hmx_lock2()   │
└────────────────────────────────────────────────────────────┘
```

---

## 3. HMX 인라인 어셈블리 상세 분석

> 코드 위치: [include/dsp/hmx_utils.h](../../../../htp-ops-lib/include/dsp/hmx_utils.h)

### 3.1 상수 정의

```c
#define HMX_FP16_TILE_N_ROWS  32       // 타일 행 수
#define HMX_FP16_TILE_N_COLS  32       // 타일 열 수
#define HMX_FP16_TILE_N_ELMS  1024     // 타일 내 요소 수 (32×32)
#define HMX_FP16_TILE_SIZE    2048     // 타일 바이트 크기 (1024 × 2 bytes)
```

### 3.2 타일 로드 (Activation + Weight)

```c
static inline void hmx_load_tiles_fp16(
    const void * restrict activation_tiles, int n_activation_tiles,
    const void * restrict weight_tiles,     int n_weight_tiles,
    int offset_bytes
)
```

**인라인 어셈블리 명령어**:

| 명령어 | 역할 | 설명 |
|--------|------|------|
| `activation.hf = mxmem(%0++#1<<12):deep` | Activation 타일 로드 | `:deep` 수식어로 VTCM에서 직접 읽기 (2048 byte 단위 이동) |
| `weight.hf = mxmem(%2++#1<<12)` | Weight 타일 로드 | Crouton 포맷의 weight 타일 연속 로드 |

`n_activation_tiles`와 `n_weight_tiles`에 따라 1~32개 타일을 순차적으로 로드하는 switch 문이 `unrolled` 형태로 구현되어 있다.

**Offset 처리**: `offset_bytes` 파라미터로 가중치 타일의 시작 위치를 조정하여 partial tile 처리가 가능하다.

### 3.3 내적 수행 (Dot Product)

```c
static inline void hmx_dot_fp16(void) {
    asm volatile("/* hmx fused dot */ \n\t" ::: "usr", "memory");
}
```

이 함수는 실제로 **명시적 명령어를 방출하지 않는다**. HMX의 내적은 `hmx_load_tiles_fp16`에서 activation과 weight 타일을 로드하는 즉시 하드웨어에 의해 **암묵적으로(implicitly) fused**되어 실행된다. `asm volatile` 배리어는 컴파일러가 명령어 순서를 재배치하지 못하도록 하는 역할만 한다.

### 3.4 Accumulator 결과 소비

```c
static inline void hmx_consume_accumulator_fp16(
    void * restrict output_tiles, int n_output_tiles
)
```

**인라인 어셈블리 명령어**:

| 명령어 | 역할 |
|--------|------|
| `cvt.hf = acc(n)` | 축적기 n번의 FP16 결과를 변환 레지스터로 이동 |
| `mxmem(%0++#1<<12) = cvt` | 변환된 결과를 VTCM 메모리에 기록 |

이 역시 1~32개 타일에 대한 unrolled switch 문으로 구현된다.

### 3.5 출력 스케일 설정

```c
// 열 스케일 초기화 (1.0f로)
static inline void hmx_init_column_scales(void) {
    asm volatile(
        "bias = mxmem2(%0) \n\t"  // 미리 1.0으로 채워진 256바이트 영역 로드
        :: "r"(hmx_init_scales_fp16) : "memory"
    );
}

// 커스텀 스케일 설정
static inline void hmx_set_output_scales(void * restrict scales) {
    asm volatile(
        "bias = mxmem2(%0) \n\t"  // 사용자 지정 스케일 로드
        :: "r"(scales) : "memory"
    );
}
```

`bias = mxmem2(...)` 명령어는 32개 열(column)에 대한 per-column 스케일을 로드한다. 이 스케일은 축적기 값을 소비할 때 자동으로 곱해진다. **MatMul에서의 활용**: 양자화 weight의 scale 값을 이 메커니즘으로 적용하여 별도의 HVX 스케일 곱셈을 생략한다.

---

## 4. HMX 자원 관리 상세

> 코드 위치: [src/dsp/hmx_mgr.c](../../../../htp-ops-lib/src/dsp/hmx_mgr.c)

### 4.1 자원 획득 절차

```
hmx_manager_setup() 호출 흐름:
  1. HAP_compute_res_acquire(SERIAL_COMPUTE_RES)
     → HMX 하드웨어 자원 전체를 OS로부터 독점 예약
  2. worker_pool_init_ex(n_workers=1, allow_hmx=1)
     → HMX 전용 워커 스레드 1개 생성
  3. hmx_init_scales_fp16 배열을 1.0f로 초기화
     → hmx_init_column_scales()에서 사용할 기본 스케일
```

### 4.2 스레드별 HMX 활성화

```c
void hmx_manager_enable_for_thread(void) {
    HAP_compute_res_hmx_lock2(hmx_res_attr, /* shared */ true);
}
```

- `shared=true`로 호출: 여러 스레드가 HMX 접근 가능 (실제 동시 사용은 스핀락으로 제어)
- 각 워커 스레드가 HMX 연산을 수행하기 전에 반드시 호출해야 함

### 4.3 HMX 스핀 락 (Mutual Exclusion)

```c
void hmx_spin_lock(void) {
    while (__builtin_expect(qurt_atomic_compare_val_and_set(
        &hmx_lock_var, 0, 1), 0) != 0) {
        // 바쁜 대기 (Busy-wait)
    }
}

void hmx_spin_unlock(void) {
    qurt_atomic_set(&hmx_lock_var, 0);
}
```

**설계 의도**: HMX 유닛은 물리적으로 하나이므로, 여러 스레드가 동시에 타일 로드/내적을 수행하면 결과가 꼬인다. 따라서:
- FlashAttention에서 각 KV 헤드별 워커가 HMX를 사용할 때 스핀 락으로 직렬화
- MatMul의 `core_dot_chunk_fp16()`에서도 HMX 사용 구간을 스핀 락으로 보호

---

## 5. 연산자에서의 HMX 사용 패턴

### 5.1 MatMul에서의 HMX 사용 흐름

> 코드 위치: [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — `core_dot_chunk_fp16()` 함수

```
1. hmx_spin_lock()                              // HMX 독점 시작
2. hmx_init_column_scales()                     // 스케일을 1.0으로 초기화
   또는 hmx_set_output_scales(scales)           // 양자화 weight의 scale 적용
3. FOR each row block (Br step):
   4. FOR each col block (Bc step):
      5. FOR each K reduction step:
         6. hmx_load_tiles_fp16(act_tiles, n_act, wt_tiles, n_wt, ...)
            // Activation과 Weight 타일을 HMX에 로드 → 자동 내적
         7. hmx_dot_fp16()
            // 컴파일러 배리어 (실제 내적은 로드에서 fused)
      8. hmx_consume_accumulator_fp16(out_tiles, n_out)
         // 축적기 결과를 VTCM 출력 영역에 저장
9. hmx_spin_unlock()                             // HMX 독점 해제
```

### 5.2 FlashAttention에서의 HMX 사용

> 코드 위치: [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c)

FlashAttention에서 HMX는 두 가지 행렬 곱에 사용된다:

| 단계 | 연산 | 좌항(Activation) | 우항(Weight) | 결과 |
|------|------|-----------------|-------------|------|
| **S = Q·K^T** | `core_dot_chunk_fp16()` | Q 타일 | K^T 타일 (vscatter로 전치) | Attention Score (S) |
| **O += P·V** | `core_mma_chunk_fp16()` | D·O_prev (대각행렬 적용) + P 타일 | V 타일 | 출력 O 갱신 |

`core_mma_chunk_fp16()`은 **축적 버전**(C += A·B)으로, 이전 축적기 값에 현재 결과를 더한다. 이를 위해:
1. Identity 타일을 로드하여 기존 O 값을 축적기에 로드
2. 새로운 P·V 결과를 기존 축적기에 더함

### 5.3 HMX 사용 시 타일 개수 최적화

> 코드 위치: [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — `find_chunk_size()` 함수

```
주어진 VTCM 영역(area_size) 내에서 최대 타일 수를 찾는 알고리즘:
1. n_act_tiles_max = min(area_size / HMX_FP16_TILE_SIZE, 32)
2. n_wt_tiles_max  = area_size / HMX_FP16_TILE_SIZE (weight 영역)
3. n_out_tiles     = n_act_tiles × n_wt_tiles  (출력 타일 수 결정)
4. n_out_tiles가 area_size에 맞을 때까지 축소
```

HMX는 한 번의 로드에 **최대 32개 타일**을 동시에 처리할 수 있으므로, activation 타일 수는 항상 32 이하로 제한된다.

---

## 6. HMX 추상화의 설계 원칙

### 6.1 Zero-Copy 원칙
- HMX 타일 로드 명령 (`mxmem`)은 **VTCM 영역의 포인터를 직접 참조**
- DDR → VTCM 전송은 별도 DMA 파이프라인으로 처리
- HMX는 VTCM만 접근하므로, 항상 VTCM 내에 데이터가 준비되어 있어야 함

### 6.2 하드웨어 직접 매핑 (Thin Abstraction)
- `hmx_utils.h`는 인라인 어셈블리를 **얇은 C 함수로 래핑**하는 수준
- 추상화 오버헤드 없이 컴파일러 인라이닝으로 최적의 성능 유지
- 타일 수에 따른 언롤링은 컴파일 타임에 결정

### 6.3 자원 생명주기 관리
```
hmx_manager_setup()         → 프로세스 시작 시 1회
hmx_manager_enable_for_thread() → 각 워커 스레드 시작 시 1회
hmx_spin_lock/unlock()      → 각 HMX 연산 블록마다
hmx_manager_teardown()      → 프로세스 종료 시 1회
```

이 계층적 자원 관리는 QuRT/HAP OS 수준의 HMX 할당과 사용자 수준의 동기화를 분리하여, 여러 연산자가 안전하게 HMX를 공유할 수 있도록 한다.
