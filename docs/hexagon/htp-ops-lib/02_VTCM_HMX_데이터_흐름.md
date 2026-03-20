# VTCM에서 HMX로의 데이터 복사 및 처리 흐름

> **분석 대상 파일들**:
> - [include/dsp/vtcm_mgr.h](../../../../htp-ops-lib/include/dsp/vtcm_mgr.h) — VTCM 메모리 관리 API
> - [src/dsp/vtcm_mgr.cc](../../../../htp-ops-lib/src/dsp/vtcm_mgr.cc) — VTCM 관리자 구현
> - [include/dsp/dma_utils.h](../../../../htp-ops-lib/include/dsp/dma_utils.h) — DMA 전송 유틸리티
> - [include/dsp/hmx_utils.h](../../../../htp-ops-lib/include/dsp/hmx_utils.h) — HMX 타일 로드 어셈블리
> - [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c) — 행렬 곱셈에서의 데이터 흐름
> - [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c) — FlashAttention에서의 데이터 흐름

---

## 1. 메모리 계층 개요

Hexagon DSP의 메모리 계층은 3단계로 구성된다:

```
┌──────────────────────────┐
│    HMX Tile Registers    │  ← 행렬 연산 레지스터 (32×32 FP16)
│     (2048 bytes/tile)    │    mxmem 명령으로 직접 로드
├──────────────────────────┤
│        VTCM (~8MB)        │  ← 벡터 밀결합 메모리
│   (Vector TCM, ~0.5 cycle)│    L1과 유사한 지연시간
├──────────────────────────┤
│      DDR (수 GB)          │  ← 주 메모리 (공유 버퍼)
│   (~100+ cycles latency) │    rpcmem으로 할당된 버퍼
└──────────────────────────┘
```

**핵심 원칙**: HMX 타일 로드 명령 (`mxmem`)은 **VTCM 주소만 참조**할 수 있다. DDR에서 직접 HMX로 데이터를 로드할 수 없으므로, 반드시 **DDR → VTCM → HMX** 경로를 따른다.

---

## 2. DDR → VTCM 데이터 전송: DMA 파이프라인

> 코드 위치: [include/dsp/dma_utils.h](../../../../htp-ops-lib/include/dsp/dma_utils.h)

### 2.1 DMA 디스크립터 구조

Hexagon DMA는 디스크립터 기반으로 동작하며, 1D/2D 전송을 지원한다:

```c
// 1D DMA 디스크립터 (연속 메모리 블록 전송)
typedef struct __attribute__((aligned(64))) {
    unsigned long long next;      // 체이닝된 다음 디스크립터
    unsigned long long src;       // 소스 주소 (DDR)
    unsigned long long dst;       // 목적지 주소 (VTCM)
    unsigned int length;          // 전송 길이 (바이트)
    // ... 추가 제어 필드
} dma_desc_1d_t;

// 2D DMA 디스크립터 (스트라이드 기반 2D 블록 전송)
typedef struct __attribute__((aligned(64))) {
    // ... 1D 필드 + 
    unsigned int src_stride;      // 소스 행 간격
    unsigned int dst_stride;      // 목적지 행 간격
    unsigned int rows;            // 전송 행 수
    unsigned int row_length;      // 행당 바이트 수
    // ...
} dma_desc_2d_t;
```

### 2.2 DMA 제어 명령어

```c
static inline void dmstart(void *desc) {
    asm volatile("dmstart(%0)" :: "r"(desc) : "memory");
}

static inline void dmlink(void *desc1, void *desc2) {
    asm volatile("dmlink(%0, %1)" :: "r"(desc1), "r"(desc2) : "memory");
}

static inline int dmpoll(void) {
    int result;
    asm volatile("%0 = dmpoll" : "=r"(result) :: "memory");
    return result;
}

static inline void dmwait(void) {
    asm volatile("dmwait" ::: "memory");
}
```

| 명령어 | 역할 |
|--------|------|
| `dmstart` | DMA 전송 시작 |
| `dmlink` | 디스크립터 체이닝 (연속 전송) |
| `dmpoll` | 전송 완료 여부 확인 (비차단) |
| `dmwait` | 전송 완료까지 대기 (차단) |

---

## 3. MatMul에서의 DDR → VTCM → HMX 흐름

> 코드 위치: [src/dsp/ops/mat_mul.c](../../../../htp-ops-lib/src/dsp/ops/mat_mul.c)

### 3.1 VTCM 영역 분할 (MatMul 전용)

```c
#define WEIGHT_AREA_SIZE     (1 * 1024 * 1024)    // 1MB
#define ACTIVATION_AREA_SIZE (1 * 1024 * 1024)    // 1MB
#define OUTPUT_AREA_SIZE     (1 * 1024 * 1024)    // 1MB
#define SCRATCH_AREA_SIZE    (1 * 1024 * 1024)    // 1MB
// 총 4MB 사용
```

```
VTCM 메모리 레이아웃 (MatMul):
┌───────────────────────────────────────┐ Base
│       WEIGHT 영역 (1MB)               │ ← DDR에서 DMA로 weight 전송
│  Pre-permuted FP16 Crouton 타일들     │
├───────────────────────────────────────┤ Base + 1MB
│       ACTIVATION 영역 (1MB)           │ ← FP32→FP16 변환 + Crouton 배치
│  Activation 타일들                     │
├───────────────────────────────────────┤ Base + 2MB
│       OUTPUT 영역 (1MB)               │ ← HMX 출력 저장
│  FP16 결과 타일들                      │
├───────────────────────────────────────┤ Base + 3MB
│       SCRATCH 영역 (1MB)              │ ← DMA 디스크립터, 임시 버퍼
│  역양자화 중간 결과 등                 │
└───────────────────────────────────────┘ Base + 4MB
```

### 3.2 FP16 Weight MatMul 데이터 흐름

> 함수: `hmx_mat_mul_permuted_w16a32()`

```
단계 1: Activation 전송 (DDR → VTCM, CPU)
  ┌──────────────────────────────────────────────┐
  │ FP32 Activation (DDR)                         │
  │ [a₀, a₁, a₂, ...] (row-major, 4 bytes each) │
  └──────────────┬───────────────────────────────┘
                 │ transfer_activation_chunk_fp32_to_fp16()
                 │  - FP32→FP16 변환 (HVX: hvx_my_wsf_to_vhf)
                 │  - Crouton 레이아웃으로 재배치
                 ▼
  ┌──────────────────────────────────────────────┐
  │ FP16 Activation Tiles (VTCM)                  │
  │ [32×32 Crouton tile₀, tile₁, ...]            │
  └──────────────────────────────────────────────┘

단계 2: Weight 전송 (DDR → VTCM, DMA)
  ┌──────────────────────────────────────────────┐
  │ FP16 Weight (DDR)                             │
  │ 이미 Pre-permuted Crouton 포맷               │
  └──────────────┬───────────────────────────────┘
                 │ DMA 1D 전송 (dmstart)
                 │  - 단순 memcpy 수준의 연속 전송
                 ▼
  ┌──────────────────────────────────────────────┐
  │ FP16 Weight Tiles (VTCM)                      │
  │ [32×32 Crouton tile₀, tile₁, ...]            │
  └──────────────────────────────────────────────┘

단계 3: HMX 내적 (VTCM → HMX Register)
  ┌─────────────────┐   ┌─────────────────┐
  │ Activation Tiles│   │ Weight Tiles    │
  │    (VTCM)       │   │   (VTCM)       │
  └────────┬────────┘   └────────┬────────┘
           │ mxmem:deep         │ mxmem
           │ (activation load)  │ (weight load)
           ▼                    ▼
  ┌────────────────────────────────────────┐
  │           HMX Accumulator              │
  │    (하드웨어 내적 수행 + 누적)         │
  └────────────────┬───────────────────────┘
                   │ cvt.hf = acc(n)
                   │ mxmem = cvt (결과 저장)
                   ▼
  ┌──────────────────────────────────────────────┐
  │ FP16 Output Tiles (VTCM OUTPUT 영역)         │
  └──────────────┬───────────────────────────────┘
                 │ FP16→FP32 변환 (HVX: hvx_my_vhf_to_wsf)
                 ▼
  ┌──────────────────────────────────────────────┐
  │ FP32 Output (DDR로 저장)                      │
  └──────────────────────────────────────────────┘
```

### 3.3 양자화 Weight MatMul 데이터 흐름 (4-Stage Pipeline)

> 함수: `hmx_mat_mul_permuted_qk_0_d16a32()`

양자화 weight(Q4_0/Q8_0/IQ4_NL)의 경우, **DMA → 역양자화 → HMX → 저장**의 4단계 파이프라인으로 동작한다:

```
시간 →
┌─────────┬──────────────┬──────────────┬──────────────┐
│ Stage 0 │  DMA load W₀ │  DMA load W₁ │  DMA load W₂ │ ...
├─────────┼──────────────┼──────────────┼──────────────┤
│ Stage 1 │              │  Dequant W₀  │  Dequant W₁  │ ...
├─────────┼──────────────┼──────────────┼──────────────┤
│ Stage 2 │              │              │  HMX dot W₀  │ ...
├─────────┼──────────────┼──────────────┼──────────────┤
│ Stage 3 │              │              │              │ Store O₀ ...
└─────────┴──────────────┴──────────────┴──────────────┘
```

**각 스테이지의 VTCM 내 더블 버퍼링**:

```
WEIGHT 영역 (1MB):
  ┌─────────────┐┌─────────────┐
  │  Buffer A    ││  Buffer B    │  ← DMA가 A에 쓸 때, HVX는 B를 역양자화
  │ (raw quants) ││ (raw quants) │
  └─────────────┘└─────────────┘

SCRATCH 영역 (1MB):
  ┌─────────────┐┌─────────────┐
  │  Buffer A    ││  Buffer B    │  ← 역양자화 결과 (FP16 Crouton)
  │ (dequant'd) ││ (dequant'd) │    HMX는 A를 읽을 때, HVX는 B에 씀
  └─────────────┘└─────────────┘
```

---

## 4. FlashAttention에서의 데이터 흐름

> 코드 위치: [src/dsp/ops/flash_attn.c](../../../../htp-ops-lib/src/dsp/ops/flash_attn.c)

### 4.1 VTCM 영역 할당 (FlashAttention, 워커당 1MB)

```
FlashAttention VTCM 레이아웃 (per-worker):
┌────────────────────────────────────┐ 시작
│  Q 타일 (Br × D/32 tiles)         │ ← DDR에서 Q 로드 + FP32→FP16 + Crouton
├────────────────────────────────────┤
│  O 타일 ×2 (더블 버퍼)            │ ← 출력 누적 결과
├────────────────────────────────────┤
│  K 타일 (Bc × D/32 tiles)         │ ← DDR에서 K 로드 + vscatter 전치
├────────────────────────────────────┤
│  V 타일 (Bc × D/32 tiles)         │ ← DDR에서 V 로드
├────────────────────────────────────┤
│  S 타일 (Br × Bc/32 tiles)        │ ← S = Q·K^T 결과 (HMX 출력)
├────────────────────────────────────┤
│  P 타일 (Br × Bc/32 tiles)        │ ← P = softmax(S) 결과
├────────────────────────────────────┤
│  D 타일 (대각 행렬)               │ ← D = diag(exp(m_prev - m_new))
├────────────────────────────────────┤
│  Column Vectors ×4                 │ ← m (rowmax), l (rowsum), etc.
├────────────────────────────────────┤
│  Row Buffers ×2                    │ ← HVX softmax 작업 버퍼
├────────────────────────────────────┤
│  HMX Scales ×2 (256B each)        │ ← per-column 스케일
└────────────────────────────────────┘ 끝 (~1MB)
```

### 4.2 K 행렬 전치 (vscatter)

K 행렬은 `(head_dim, seq_len)` 형태이지만, HMX는 행 방향으로 activation을 읽으므로 `(seq_len, head_dim)` 방향의 열-우선(column-major) Crouton 레이아웃이 필요하다.

```c
// src/dsp/ops/flash_attn.c 내 K 로딩 코드:
// vscatter 명령어로 FP16 K를 Crouton 열-우선 배치로 전치
for (각 K 행) {
    HVX_Vector k_row = *(HVX_Vector *)k_src;      // DDR에서 HVX 벡터 로드
    Q6_vscatter_RMVhW(k_vtcm, stride, offsets, k_row);  // VTCM에 산포 저장
}
```

**vscatter**는 HVX 벡터의 각 요소를 임의의 VTCM 주소에 분산 저장하는 명령어로, 전치를 메모리 복사 없이 수행한다.

### 4.3 Q 행렬 전송 (FP32 → FP16 Crouton)

```
DDR (FP32)                        VTCM (FP16 Crouton)
┌────────────────────┐            ┌────────────────────┐
│ q₀₀ q₀₁ q₀₂ ...   │  FP32→FP16 │ ┌──32×32 Tile───┐ │
│ q₁₀ q₁₁ q₁₂ ...   │ ─────────▶ │ │ q₀₀ q₁₀ ...  │ │
│ ...                │  + repack  │ │ q₀₁ q₁₁ ...  │ │
└────────────────────┘            │ └────────────────┘ │
                                  └────────────────────┘
```

### 4.4 FlashAttention 전체 데이터 흐름

```
FOR each KV block (j):
  ┌─────────────────────────────────────────────────┐
  │ 1. K[j] 로드: DDR → vscatter → VTCM (전치)      │
  │ 2. V[j] 로드: DDR → VTCM (컬럼 메이저)           │
  ├─────────────────────────────────────────────────┤
  │ 3. S = Q × K^T: VTCM(Q) + VTCM(K) → HMX → VTCM(S) │
  │    └ hmx_load_tiles_fp16() + hmx_consume_accumulator │
  ├─────────────────────────────────────────────────┤
  │ 4. Softmax(S→P): VTCM(S) → HVX 처리 → VTCM(P)     │
  │    └ rowmax, exp2(S-rowmax), rowsum 계산             │
  │    └ vgather 기반 exp2 테이블 룩업 (VTCM에 사전 계산)│
  ├─────────────────────────────────────────────────┤
  │ 5. O += P × V: VTCM(P) + VTCM(V) → HMX → VTCM(O)  │
  │    └ D·O_prev + P·V (대각 스케일링 + 행렬곱 누적)   │
  └─────────────────────────────────────────────────┘
```

---

## 5. 캐시 일관성 관리

> 코드 위치: [src/dsp/op_executor.cc](../../../../htp-ops-lib/src/dsp/op_executor.cc)

DDR 버퍼는 CPU와 DSP 간에 공유되므로, 캐시 일관성 관리가 필수적이다:

```c
// 입력 버퍼: D-cache 무효화 (DDR에서 최신 데이터 읽기 보장)
static void validate_in_bufs(const RpcmemBufAddr * bufs, int n) {
    for (int i = 0; i < n; i++) {
        qurt_mem_cache_clean((qurt_addr_t)addr, size,
                             QURT_MEM_CACHE_INVALIDATE);
    }
}

// 출력 버퍼: D-cache 플러시 (DSP 결과를 DDR에 기록 보장)
static void validate_out_bufs(const RpcmemBufAddr * bufs, int n) {
    for (int i = 0; i < n; i++) {
        qurt_mem_cache_clean((qurt_addr_t)addr, size,
                             QURT_MEM_CACHE_FLUSH);
    }
}
```

**양자화 Weight 특수 처리**: 양자화 weight 버퍼는 첫 번째 호출 시에만 cache invalidate하고 이후에는 "uncached" 모드(`false`)로 처리하여 반복적인 cache 관리 오버헤드를 제거한다.

---

## 6. 핵심 요약: VTCM 중심의 데이터 흐름

```
DDR (rpcmem)                 VTCM (8MB)              HMX
┌───────────────┐      ┌─────────────────┐     ┌──────────┐
│ FP32 Activation│─DMA→│ FP16 Crouton    │─mxmem→│ Act Tile │
│               │ +HVX │ Activation Tiles│       └────┬─────┘
├───────────────┤      ├─────────────────┤            │
│ FP16 Weight   │─DMA─→│ FP16 Crouton   │─mxmem────→│ Wt Tile │
│ (pre-permuted)│      │ Weight Tiles    │            └────┬─────┘
├───────────────┤      ├─────────────────┤                 │
│ Q4_0 Weight   │─DMA─→│ Raw Quants     │                 │
│ (quantized)   │      │   ↓ HVX dequant│                 ▼
│               │      │ FP16 Crouton   │─mxmem→ ┌──────────────┐
│               │      │ Weight Tiles   │        │ Accumulator  │
├───────────────┤      ├─────────────────┤       │ (FP16 dot)   │
│ FP32 Output   │←DMA──│ FP16 Output    │←mxmem──│  ↓ cvt.hf    │
│ (result)      │ +HVX │ Tiles          │        └──────────────┘
└───────────────┘      └─────────────────┘

범례:
  DMA   = Hexagon DMA 엔진 (비동기 전송)
  HVX   = Hexagon Vector eXtension (변환/역양자화)
  mxmem = HMX 타일 로드/저장 명령어
```

VTCM은 DDR과 HMX 사이의 **고속 스테이징 영역**으로 기능하며, 모든 데이터는 반드시 VTCM을 경유하여 HMX로 전달된다. 이 아키텍처로 인해 VTCM의 제한된 크기(~8MB) 내에서 효율적인 타일링 전략(Topic 5에서 상세 설명)이 필수적이다.
