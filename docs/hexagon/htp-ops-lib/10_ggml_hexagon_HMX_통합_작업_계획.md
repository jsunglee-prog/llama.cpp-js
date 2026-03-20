# 10. 기존 ggml-hexagon에 htp-ops-lib HMX 기능 통합 작업 계획

> 기존 ggml-hexagon 백엔드의 HVX 25+ 연산을 유지하면서, htp-ops-lib의 HMX 커널 3개를 단일 skel에 통합하는 계획

## 1. 통합 목표

```
현재 상태:
  ggml-hexagon (libggml-htp-v73.so)  →  HVX 25+ 연산
  htp-ops-lib  (libhtp_ops_skel.so)  →  HMX 3개 연산 (별도 바이너리)

목표 상태:
  ggml-hexagon (libggml-htp-v73.so)  →  HVX 25+ 연산 + HMX 3개 연산 (단일 바이너리)
```

단일 skel로 통합하면:
- VTCM 8MB를 하나의 관리자로 제어 → 자원 충돌 없음
- HMX/HVX 전환 시 VTCM 재할당 불필요
- 배포 시 skel 하나만 관리

## 2. 전체 아키텍처

```
┌────────────────────────── Host (Android CPU) ──────────────────────────┐
│                                                                        │
│  ggml-hexagon.cpp                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ supports_op()                                                   │   │
│  │  ├── 큰 MUL_MAT (K%32==0, N%32==0)  → HMX 경로 선택          │   │
│  │  ├── 기존 MUL_MAT                    → HVX 경로 유지          │   │
│  │  ├── FLASH_ATTN_EXT (FP16 KV)        → HMX 경로 선택          │   │
│  │  └── 나머지 25+ ops                  → 기존 HVX 경로          │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ graph_compute()                                                 │   │
│  │  ├── init_hmx_matmul_req()    → HTP_OP_MUL_MAT_HMX            │   │
│  │  ├── init_hmx_flash_attn_req() → HTP_OP_FLASH_ATTN_EXT_HMX    │   │
│  │  ├── init_binary_req()         → HTP_OP_MUL_MAT (기존)        │   │
│  │  └── init_*_req()              → HTP_OP_* (기존)              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ ggml_hexagon_dispatch_op<init_req>()                            │   │
│  │  └── dspqueue_write() ─────────────────────────────────┐       │   │
│  └─────────────────────────────────────────────────────────│───────┘   │
│                                                            │           │
├────────────────────────── dspqueue ────────────────────────┼───────────┤
│                                                            │           │
│  ┌──────── DSP (Hexagon cDSP, libggml-htp-v73.so) ────────┼───────┐  │
│  │                                                         ▼       │  │
│  │  htp_packet_callback() → switch(req.op)                         │  │
│  │  ┌──────────────────────────────────────────────────────────┐   │  │
│  │  │ case HTP_OP_MUL_MAT:                                    │   │  │
│  │  │   vtcm_acquire() → op_matmul() [HVX] → vtcm_release()  │   │  │
│  │  ├──────────────────────────────────────────────────────────┤   │  │
│  │  │ case HTP_OP_MUL_MAT_HMX:            ← 신규 추가        │   │  │
│  │  │   vtcm_acquire() → op_matmul_hmx() [HMX] → vtcm_release│   │  │
│  │  ├──────────────────────────────────────────────────────────┤   │  │
│  │  │ case HTP_OP_FLASH_ATTN_EXT:                              │   │  │
│  │  │   vtcm_acquire() → op_flash_attn_ext() [HVX] → release  │   │  │
│  │  ├──────────────────────────────────────────────────────────┤   │  │
│  │  │ case HTP_OP_FLASH_ATTN_EXT_HMX:     ← 신규 추가        │   │  │
│  │  │   vtcm_acquire() → op_flash_attn_hmx() [HMX] → release  │   │  │
│  │  ├──────────────────────────────────────────────────────────┤   │  │
│  │  │ case HTP_OP_ADD / MUL / ROPE / ...: (기존 HVX 그대로)   │   │  │
│  │  └──────────────────────────────────────────────────────────┘   │  │
│  │                                                                 │  │
│  │  자원 관리 (통합)                                               │  │
│  │  ┌──────────────────────────────────────────────────┐          │  │
│  │  │ vtcm_alloc():  8MB 전체 확보 + HMX 번들 포함    │          │  │
│  │  │ worker_pool:   HVX 6스레드 (기존)               │          │  │
│  │  │ hmx_worker:    HMX 1스레드 (신규)               │          │  │
│  │  │ exp2_table:    VTCM 256KB 상주 (신규)           │          │  │
│  │  └──────────────────────────────────────────────────┘          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

## 3. HVX/HMX 공존이 가능한 이유

### 3.1 VTCM은 이미 HMX를 포함

기존 ggml-hexagon의 `vtcm_alloc()`이 **이미 HMX 자원도 함께 요청**하고 있습니다:

**파일**: `ggml/src/ggml-hexagon/htp/main.c` (line 202-237)

```c
static int vtcm_alloc(struct htp_context * ctx) {
    unsigned int vtcm_size = 8 * 1024 * 1024;  // 8MB
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, 0, vtcm_size);
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (void *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);  // ← HMX도 함께 요청!

    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000);
    ...
}
```

따라서 **별도의 HMX 자원 요청 없이**, 기존 `vtcm_acquire()/vtcm_release()` 사이에서 HMX 명령어를 자유롭게 사용할 수 있습니다.

### 3.2 기존 연산 패턴과 동일한 흐름

모든 기존 op은 동일한 패턴을 따릅니다:

```c
vtcm_acquire(ctx);       // VTCM + HMX 잠금
rsp_status = op_xxx();   // HVX 또는 HMX 커널 실행
vtcm_release(ctx);       // VTCM + HMX 해제
```

새로운 HMX op도 이 패턴을 그대로 따르면 됩니다.

### 3.3 htp-ops-lib의 별도 매니저가 불필요

| htp-ops-lib 매니저 | 통합 시 대체 | 이유 |
|---------------------|-------------|------|
| `vtcm_manager_setup()` | **제거** — `ctx->vtcm_base` 사용 | 기존 vtcm_alloc()이 이미 확보 |
| `hmx_manager_setup()` | **제거** — `vtcm_alloc`에 HMX 포함 | `HAP_compute_res_attr_set_hmx_param` |
| `power_setup()` | **제거** — `htp_iface_open()`에서 수행 | 기존 코드가 HVX+HMX 모두 power up |
| `worker_pool` (6스레드) | 기존 `ctx->worker_pool` 사용 | 동일한 구조 |
| `hmx_worker_pool` (1스레드) | 신규 `ctx->hmx_worker_pool` 추가 | HMX 전용 스레드 |

## 4. 5단계 통합 작업

### 4.1 — 1단계: HMX 커널 코드를 skel에 병합

htp-ops-lib의 DSP 커널 소스를 ggml-hexagon skel 빌드에 추가합니다.

**복사 및 리팩토링 대상:**

| 원본 (htp-ops-lib) | 대상 (ggml-hexagon/htp/) | 줄 수 | 리팩토링 내용 |
|---------------------|--------------------------|-------|---------------|
| `src/dsp/ops/mat_mul.c` | `hmx-matmul-ops.c` | 1,452 | vtcm_base를 파라미터로 변경 |
| `src/dsp/ops/flash_attn.c` | `hmx-flash-attn-ops.c` | 1,591 | vtcm_base를 파라미터로 변경 |
| `src/dsp/ops/rms_norm.c` | `hmx-rms-norm-ops.c` | ~200 | 동일 |
| `src/dsp/ops/precompute_table.c` | `hmx-precompute-table.c` | ~100 | 동일 |
| `include/dsp/hmx_utils.h` | `hmx-utils.h` | ~150 | 그대로 복사 (인라인 어셈블리) |

**리팩토링 핵심:** htp-ops-lib 커널은 글로벌 `vtcm_manager::vtcm_base`를 참조하므로, 이를 파라미터로 전달하도록 변경:

```c
// Before (htp-ops-lib 원본):
void * base = vtcm_manager_get_vtcm_base();

// After (통합 버전):
void * base = octx->ctx->vtcm_base;  // htp_context에서 직접 접근
```

**CMakeLists.txt 변경:**

```cmake
# ggml/src/ggml-hexagon/htp/CMakeLists.txt
add_library(${HTP_LIB} SHARED
    # 기존 HVX ops (변경 없음)
    main.c
    htp_iface_skel.c
    worker-pool.c
    hex-dma.c
    matmul-ops.c
    binary-ops.c
    unary-ops.c
    sum-rows-ops.c
    softmax-ops.c
    act-ops.c
    rope-ops.c
    flash-attn-ops.c
    set-rows-ops.c
    get-rows-ops.c
    cpy-ops.c
    argsort-ops.c
    ssm-conv.c
    # HMX ops (신규 추가)
    hmx-matmul-ops.c
    hmx-flash-attn-ops.c
    hmx-rms-norm-ops.c
    hmx-precompute-table.c
)
```

### 4.2 — 2단계: VTCM 통합 관리자 구현

기존 `htp_context`에 HMX 관련 필드를 추가합니다.

**파일**: `ggml/src/ggml-hexagon/htp/htp-ctx.h`

```c
struct htp_context {
    // 기존 필드 (변경 없음)
    dspqueue_t            queue;
    dma_queue *           dma[HTP_MAX_NTHREADS];
    worker_pool_context_t worker_pool;
    uint32_t              n_threads;
    
    int thread_id;
    int thread_prio;
    
    uint8_t * vtcm_base;
    size_t    vtcm_size;
    uint32_t  vtcm_rctx;
    
    atomic_bool vtcm_valid;
    atomic_bool vtcm_inuse;
    atomic_bool vtcm_needs_release;
    
    uint32_t opmask;
    
    // HMX 관련 추가 필드
    worker_pool_context_t hmx_worker_pool;   // HMX 전용 worker (1 스레드)
    uint8_t *             exp2_table;        // exp2 lookup table (256KB, VTCM 상주)
    bool                  hmx_initialized;   // HMX 초기화 완료 여부
};
```

**VTCM 레이아웃 설계:**

```
VTCM 8MB (0x800000):
┌─────────────────────────────────────────┐  0x000000
│  exp2 lookup table (256KB, 상주)        │
├─────────────────────────────────────────┤  0x040000
│                                         │
│  작업 영역 (7.75MB)                     │
│  - HVX op 실행 시: 전체 사용            │
│  - HMX matmul 시: 4×1MB 타일링          │
│  - HMX flash_attn 시: 6×1MB 워커 할당   │
│                                         │
└─────────────────────────────────────────┘  0x800000
```

**main.c의 `htp_iface_start()`에 HMX 초기화 추가:**

```c
AEEResult htp_iface_start(remote_handle64 handle, ...) {
    ...
    // 기존: VTCM 할당 + HVX worker pool
    vtcm_alloc(ctx);  // 이미 HMX 포함
    worker_pool_init(&ctx->worker_pool, n_hvx);

    // 신규: HMX worker pool + exp2 table
    worker_pool_init_ex(&ctx->hmx_worker_pool, 8192, 1, 1);
    ctx->exp2_table = ctx->vtcm_base;  // VTCM 시작 256KB
    precompute_exp2_table(ctx->exp2_table);
    ctx->hmx_initialized = true;
    ...
}
```

### 4.3 — 3단계: 가중치 Crouton Repack 처리

HMX 커널은 가중치를 32×32 Crouton 타일 레이아웃으로 변환해야 합니다.

**Crouton 레이아웃 인덱싱:**

```
원본 (row-major): data[row * cols + col]
Crouton:          data[(row & ~1) * 32 + col * 2 + (row & 1)]
```

**두 가지 접근법:**

#### 옵션 A: 호스트에서 사전 변환 (권장)

모델 로딩 시 CPU에서 한 번만 변환하여 `dspqueue_buffer`에 저장:

```cpp
// ggml-hexagon.cpp — set_tensor 또는 init_tensor 시점
static void ggml_hexagon_prepare_hmx_weight(ggml_tensor * weight) {
    if (!should_use_hmx(weight)) return;
    
    size_t K = weight->ne[0], N = weight->ne[1];
    // FP16으로 변환 + Crouton repack
    ggml_fp16_t * repacked = crouton_repack_fp16(weight->data, K, N);
    memcpy(weight->data, repacked, K * N * sizeof(ggml_fp16_t));
    free(repacked);
}
```

장점: DSP 커널 수정 최소화, 반복 실행 시 변환 비용 0

#### 옵션 B: DSP에서 on-the-fly 변환

htp-ops-lib의 `mat_mul.c`에 이미 DMA + repack 파이프라인이 구현되어 있음:

```c
// mat_mul.c의 기존 로직
// DDR → DMA → VTCM 버퍼에 로드하면서 Crouton 형태로 재배치
dma_copy_weight_to_vtcm_crouton(weight_ddr, vtcm_weight, K, N);
```

장점: 호스트 변경 불필요, 기존 ggml 텐서 포맷 유지

**양자화 타입별 처리:**

| 타입 | HMX 커널 내부 처리 |
|------|---------------------|
| F16 | Crouton repack만 필요 |
| Q4_0 | VLUT16으로 FP16 역양자화 → Crouton (htp-ops-lib에 구현됨) |
| Q8_0 | VLUT16으로 FP16 역양자화 → Crouton (htp-ops-lib에 구현됨) |
| IQ4_NL | 비균일 VLUT16 테이블 → Crouton (htp-ops-lib에 구현됨) |

### 4.4 — 4단계: 호스트 측 디스패치 로직 추가

#### (1) htp-msg.h — op enum 확장

```c
// ggml/src/ggml-hexagon/htp/htp-msg.h
enum htp_op {
    // 기존 (변경 없음)
    HTP_OP_MUL = 0,
    HTP_OP_ADD = 1,
    ...
    HTP_OP_SSM_CONV,
    
    // HMX ops (신규 추가)
    HTP_OP_MUL_MAT_HMX,              // HMX FP16/Q4_0/Q8_0/IQ4_NL matmul
    HTP_OP_FLASH_ATTN_EXT_HMX,       // HMX flash attention
    HTP_OP_RMS_NORM_HMX,             // HMX rms norm (향후 활성화)
    
    INVALID
};
```

#### (2) htp-ops.h — 커널 함수 선언 추가

```c
// ggml/src/ggml-hexagon/htp/htp-ops.h
// 기존 함수들 유지
int op_matmul(struct htp_ops_context * octx);
int op_flash_attn_ext(struct htp_ops_context * octx);
...

// HMX 커널 함수 (신규)
int op_matmul_hmx(struct htp_ops_context * octx);
int op_flash_attn_ext_hmx(struct htp_ops_context * octx);
int op_rms_norm_hmx(struct htp_ops_context * octx);
```

#### (3) main.c — 패킷 콜백에 HMX case 추가

```c
// ggml/src/ggml-hexagon/htp/main.c — htp_packet_callback() 내 switch
case HTP_OP_MUL_MAT_HMX:
    if (n_bufs != 3) {
        FARF(ERROR, "Bad hmx-matmul-req buffer list");
        continue;
    }
    proc_matmul_hmx_req(ctx, &req, bufs, n_bufs);
    break;

case HTP_OP_FLASH_ATTN_EXT_HMX:
    if (!(n_bufs >= 4 && n_bufs <= 6)) {
        FARF(ERROR, "Bad hmx-flash-attn-ext-req buffer list");
        continue;
    }
    proc_flash_attn_ext_hmx_req(ctx, &req, bufs, n_bufs);
    break;
```

**proc_matmul_hmx_req 구현 (proc_matmul_req 패턴 복제):**

```c
static void proc_matmul_hmx_req(struct htp_context * ctx,
                                 struct htp_general_req * req,
                                 struct dspqueue_buffer * bufs,
                                 size_t n_bufs) {
    struct dspqueue_buffer rsp_bufs[1];
    rsp_bufs[0] = bufs[2];
    rsp_bufs[0].flags = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                         DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);

    struct htp_ops_context octx = { 0 };
    octx.ctx       = ctx;
    octx.src0      = req->src0;
    octx.src1      = req->src1;
    octx.dst       = req->dst;
    octx.flags     = req->flags;
    octx.op        = req->op;
    octx.src0.data = (uint32_t) bufs[0].ptr;
    octx.src1.data = (uint32_t) bufs[1].ptr;
    octx.dst.data  = (uint32_t) bufs[2].ptr;
    octx.n_threads = ctx->n_threads;

    struct profile_data prof;
    profile_start(&prof);

    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        rsp_status = op_matmul_hmx(&octx);  // HMX 커널 호출
        vtcm_release(ctx);
    }

    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, 1, &prof);
}
```

#### (4) ggml-hexagon.cpp — HMX 사용 조건 판단

```cpp
// ggml-hexagon.cpp
static bool ggml_hexagon_should_use_hmx_matmul(ggml_hexagon_session * sess,
                                                const struct ggml_tensor * op) {
    auto * weight = op->src[0];
    auto * act    = op->src[1];

    // 타입 조건: HMX가 지원하는 가중치 타입
    bool type_ok = (weight->type == GGML_TYPE_F16  ||
                    weight->type == GGML_TYPE_Q4_0 ||
                    weight->type == GGML_TYPE_Q8_0 ||
                    weight->type == GGML_TYPE_IQ4_NL);

    // 형상 조건: HMX 32×32 타일에 정렬
    bool shape_ok = (weight->ne[0] % 32 == 0) &&
                    (weight->ne[1] % 32 == 0) &&
                    ggml_nrows(op) == op->ne[1] &&
                    ggml_nrows(act) == act->ne[1];

    // 크기 조건: 충분히 커야 HMX 오버헤드 상쇄
    bool big_enough = (weight->ne[0] * weight->ne[1]) >= (1024 * 1024);

    return type_ok && shape_ok && big_enough;
}

static bool ggml_hexagon_should_use_hmx_flash_attn(ggml_hexagon_session * sess,
                                                     const struct ggml_tensor * op) {
    auto * q    = op->src[0];
    auto * k    = op->src[1];
    auto * v    = op->src[2];
    auto * mask = op->src[3];

    float max_bias      = *(float *)&op->op_params[1];
    float logit_softcap = *(float *)&op->op_params[2];

    return op->type == GGML_TYPE_F32 &&
           q->type  == GGML_TYPE_F32 &&
           k->type  == GGML_TYPE_F16 &&
           v->type  == GGML_TYPE_F16 &&
           mask && mask->type == GGML_TYPE_F16 &&
           max_bias == 0 && logit_softcap == 0;
}
```

#### (5) ggml-hexagon.cpp — init_req 및 graph_compute 분기

```cpp
// init_req 함수
static inline size_t init_hmx_matmul_req(htp_general_req * req,
                                          dspqueue_buffer * bufs,
                                          const ggml_tensor * t) {
    req->op = HTP_OP_MUL_MAT_HMX;

    size_t n_bufs = 0;
    n_bufs += htp_req_buff_init(&req->src0, &bufs[n_bufs], t->src[0],
                                 DSPQBUF_TYPE_CONSTANT);
    n_bufs += htp_req_buff_init(&req->src1, &bufs[n_bufs], t->src[1],
                                 DSPQBUF_TYPE_CPU_WRITE_DSP_READ);
    n_bufs += htp_req_buff_init(&req->dst,  &bufs[n_bufs], t,
                                 DSPQBUF_TYPE_DSP_WRITE_CPU_READ);
    return n_bufs;
}

// graph_compute에서 분기
case GGML_OP_MUL_MAT:
    if (ggml_hexagon_should_use_hmx_matmul(sess, node)) {
        ggml_hexagon_dispatch_op<init_hmx_matmul_req>(sess, node, flags);
    } else if (src0_is_constant) {
        ggml_hexagon_dispatch_op<init_binary_req<true>>(sess, node, flags);
    } else {
        ggml_hexagon_dispatch_op<init_binary_req<false>>(sess, node, flags);
    }
    break;

case GGML_OP_FLASH_ATTN_EXT:
    if (ggml_hexagon_should_use_hmx_flash_attn(sess, node)) {
        ggml_hexagon_dispatch_op<init_hmx_flash_attn_req>(sess, node, flags);
    } else {
        ggml_hexagon_dispatch_op<init_flash_attn_ext_req>(sess, node, flags);
    }
    break;
```

### 4.5 — 5단계: worker_pool 통합

**기존 vs 신규:**

```
기존 ggml-hexagon:
  ctx->worker_pool  → HVX 최대 6스레드 (n_hvx로 설정)
  
통합 후:
  ctx->worker_pool      → HVX 최대 6스레드 (기존 유지)
  ctx->hmx_worker_pool  → HMX 전용 1스레드 (신규)
```

htp-ops-lib의 HMX 커널은 HMX 유닛이 1개이므로 1스레드만 필요. 나머지 HVX 스레드는 DMA 전송이나 역양자화 전처리에 사용 가능.

**htp_iface_start()에 추가:**

```c
// main.c — htp_iface_start() 내
ctx->n_threads = n_hvx;
worker_pool_init(&ctx->worker_pool, n_hvx);

// HMX worker pool (1 스레드, stack 8KB)
int hmx_err = worker_pool_init_ex(&ctx->hmx_worker_pool, 8192, 1, 1);
if (hmx_err) {
    FARF(ERROR, "HMX worker pool init failed");
}

// exp2 lookup table 사전 계산 (FlashAttention에 필요)
ctx->exp2_table = ctx->vtcm_base;
precompute_exp2_table_f16(ctx->exp2_table, 256 * 1024);
ctx->hmx_initialized = true;
```

**htp_iface_stop()에 정리 추가:**

```c
// main.c — htp_iface_stop() 내
if (ctx->hmx_worker_pool) {
    worker_pool_release(&ctx->hmx_worker_pool);
}
vtcm_free(ctx);
```

## 5. 수정 파일 체크리스트

| # | 작업 | 대상 파일 | 난이도 | 비고 |
|---|------|-----------|--------|------|
| 1 | `htp_op` enum에 HMX op 코드 추가 | `htp/htp-msg.h` | 쉬움 | 3개 enum 추가 |
| 2 | HMX 커널 함수 선언 추가 | `htp/htp-ops.h` | 쉬움 | 3개 함수 선언 |
| 3 | htp-ops-lib 커널 소스 복사+리팩토링 | `htp/hmx-*.c` (신규) | **어려움** | ~3,000줄 포팅, VTCM 파라미터화 |
| 4 | VTCM 매니저 통합 (글로벌 → 파라미터) | HMX 커널 내부 | 중간 | `vtcm_manager_get_vtcm_base()` → `octx->ctx->vtcm_base` |
| 5 | `htp_context`에 HMX 필드 추가 | `htp/htp-ctx.h` | 쉬움 | 3개 필드 |
| 6 | `htp_iface_start/stop`에 HMX 초기화/정리 | `htp/main.c` | 중간 | worker_pool, exp2_table |
| 7 | `htp_packet_callback` switch case 추가 | `htp/main.c` | 쉬움 | `proc_matmul_req` 패턴 복제 |
| 8 | `proc_*_hmx_req` 래퍼 함수 추가 | `htp/main.c` | 쉬움 | 기존 proc 함수 패턴과 동일 |
| 9 | `supports_op`에 HMX 판단 로직 추가 | `ggml-hexagon.cpp` | 중간 | 타입/크기/형상 조건 |
| 10 | `init_hmx_*_req` 함수 작성 | `ggml-hexagon.cpp` | 쉬움 | 기존 init_req 패턴 |
| 11 | `graph_compute`에 HMX 분기 추가 | `ggml-hexagon.cpp` | 쉬움 | if/else 추가 |
| 12 | Crouton repack 처리 | `ggml-hexagon.cpp` 또는 커널 | **어려움** | 옵션 A/B 중 선택 |
| 13 | CMakeLists.txt에 소스 추가 | `htp/CMakeLists.txt` | 쉬움 | 파일 목록 추가 |
| 14 | HMX 인라인 어셈블리 헤더 복사 | `htp/hmx-utils.h` (신규) | 쉬움 | 그대로 복사 |
| 15 | exp2 lookup table VTCM 상주 설정 | `htp/main.c` | 쉬움 | `precompute_table.c` 호출 |

## 6. 성능 기대치

### HVX → HMX 전환 시 예상 이점

| 연산 | HVX (기존) | HMX (통합 후) | 예상 개선 |
|------|-----------|--------------|-----------|
| MUL_MAT (4096×4096, FP16) | ~200 GFLOPS | ~2,000 GFLOPS | **~10x** |
| MUL_MAT (4096×4096, Q4_0) | ~400 GFLOPS (dequant+dot) | ~1,500 GFLOPS (VLUT16+HMX) | **~4x** |
| FLASH_ATTN (D=128, KV=2K) | HVX dot product | HMX 32×32 타일 곱 | **~5-8x** |
| RMS_NORM | HVX SIMD | 변화 없음 (현재 비활성화) | - |

> 수치는 Snapdragon 8 Gen 3 (v73) 기준 추정치. 실측 필요.

### HVX 연산은 영향 없음

| 기존 HVX 연산 (25+) | 변화 |
|----------------------|------|
| ADD, MUL, SUB, DIV | 변화 없음 |
| SOFTMAX, SILU, GELU | 변화 없음 |
| ROPE, SET_ROWS, GET_ROWS | 변화 없음 |
| CPY, ARGSORT, SSM_CONV | 변화 없음 |
| SUM_ROWS, SCALE, SQR, SQRT | 변화 없음 |
| 작은 MUL_MAT (HMX 조건 미충족) | 기존 HVX 경로 유지 |
| FLASH_ATTN_EXT (FP32 KV) | 기존 HVX 경로 유지 |

## 7. 위험 요소 및 주의사항

### 7.1 Crouton 포맷 호환성

htp-ops-lib의 `block_q4_0` QK_K=256과 ggml의 `block_q4_0` QK=32는 **서로 다른 구조체**입니다. HMX 커널 포팅 시 ggml의 표준 `block_q4_0`에서 직접 역양자화하도록 수정하거나, 변환 레이어를 추가해야 합니다.

### 7.2 VTCM 공간 경합

exp2 table(256KB)을 VTCM에 상주시키면 HVX op이 사용 가능한 VTCM이 7.75MB로 줄어듭니다. 대부분의 HVX op은 전체 8MB를 사용하지 않으므로 문제없으나, 확인이 필요합니다.

### 7.3 빌드 환경

HMX 인라인 어셈블리(`hmx_utils.h`)는 Hexagon SDK 6.x의 `hexagon-clang`이 필요합니다. 기존 ggml-hexagon skel 빌드도 동일 컴파일러를 사용하므로 호환성 문제는 없을 것으로 예상됩니다.

### 7.4 테스트 전략

```
1차: 단위 테스트
  - HMX matmul 커널만 독립 실행 (기존 htp_ops_test 활용)
  - 기존 HVX 연산 회귀 테스트

2차: 통합 테스트
  - 작은 모델 (TinyLlama 1.1B)로 전체 추론
  - HVX 전용 vs HVX+HMX 성능 비교
  - VTCM 경합 시나리오 (큰 KV cache + 큰 행렬곱)

3차: 벤치마크
  - Snapdragon 8 Gen 3 디바이스에서 tok/s 측정
  - Prefill (큰 배치) vs Decode (배치=1) 성능 프로파일링
```

## 8. 예상 작업량

| 단계 | 작업 | 예상 소요 |
|------|------|----------|
| 1단계 | 커널 소스 복사 + 리팩토링 | 1주 |
| 2단계 | VTCM 통합 매니저 | 3일 |
| 3단계 | Crouton repack | 3-5일 |
| 4단계 | 호스트 디스패치 | 2일 |
| 5단계 | worker_pool + 빌드 | 1일 |
| 테스트 | 단위/통합/벤치마크 | 1주 |
| **합계** | | **~2-3주** |

가장 어려운 부분은 **1단계 커널 리팩토링** (VTCM 글로벌 → 파라미터)과 **3단계 Crouton repack 전략 결정**입니다.
