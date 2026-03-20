# DSP 측 (HTP skel) Op 처리 메커니즘 상세 분석

> 분석 파일: `ggml/src/ggml-hexagon/htp/` 디렉토리

---

## 1. 세션 초기화

### IDL 인터페이스

**파일**: [htp/htp_iface.idl](../../ggml/src/ggml-hexagon/htp/htp_iface.idl)

```idl
interface htp_iface : remote_handle64 {
    AEEResult start(in uint32 sess_id, in uint64 dsp_queue_id, in uint32 n_hvx);
    AEEResult stop();
    AEEResult enable_etm();
    AEEResult disable_etm();
};
```

### `htp_iface_open()` — [main.c L29~L100](../../ggml/src/ggml-hexagon/htp/main.c#L29-L100)

1. `htp_context` 구조체 할당
2. FARF 로깅 설정
3. **HVX 전원 ON** (`HAP_power_set` → `HAP_power_set_HVX`)
4. **HMX 전원 ON** (`HAP_power_set` → `HAP_power_set_HMX`) — [L91~L100](../../ggml/src/ggml-hexagon/htp/main.c#L91-L100)
5. DCVS 성능 모드 설정 (MAX corner)

> **중요**: HMX 전원은 이미 `open` 시점에 켜짐. HMX Op 추가 시 별도 전원 관리 불필요.

### `htp_iface_start()` — [main.c L253~L307](../../ggml/src/ggml-hexagon/htp/main.c#L253-L307)

1. `dspqueue_import()` — CPU에서 생성한 큐를 DSP에 import
2. VTCM 할당 (`vtcm_alloc()`)
3. HVX 스레드 수 결정
4. DMA 큐 생성 (스레드당 1개)
5. Worker Pool 초기화 (`worker_pool_init()`)

---

## 2. 핵심 컨텍스트 구조체

### `htp_context` — [htp/htp-ctx.h](../../ggml/src/ggml-hexagon/htp/htp-ctx.h)

```c
struct htp_context {
    dspqueue_t            queue;
    dma_queue *           dma[HTP_MAX_NTHREADS]; // 스레드당 DMA 큐
    worker_pool_context_t worker_pool;             // 멀티스레드 워커 풀
    uint32_t              n_threads;               // HVX 스레드 수

    uint8_t * vtcm_base;                           // VTCM 기본 주소
    size_t    vtcm_size;                           // VTCM 크기 (보통 8MB)
    uint32_t  vtcm_rctx;                           // compute resource 핸들

    atomic_bool vtcm_valid;
    atomic_bool vtcm_inuse;
    atomic_bool vtcm_needs_release;
};
```

### `htp_ops_context` — [htp/htp-ops.h](../../ggml/src/ggml-hexagon/htp/htp-ops.h)

```c
struct htp_ops_context {
    struct htp_context * ctx;       // DSP 컨텍스트 (VTCM, worker pool 등)

    enum htp_op op;                 // HTP Op 코드
    int32_t     op_params[16];      // Op 파라미터 (epsilon 등)

    struct htp_tensor src0;         // 입력 텐서들
    struct htp_tensor src1;
    struct htp_tensor src2;
    struct htp_tensor src3;
    struct htp_tensor src4;
    struct htp_tensor dst;          // 출력 텐서

    struct htp_spad src0_spad;      // VTCM 스크래치패드 (Op이 직접 설정)
    struct htp_spad src1_spad;
    struct htp_spad src2_spad;
    struct htp_spad src3_spad;
    struct htp_spad dst_spad;

    worker_pool_context_t * wpool;
    uint32_t                n_threads;
    uint32_t                flags;
};
```

---

## 3. 패킷 콜백 (`htp_packet_callback`)

**위치**: [main.c L1010~L1200](../../ggml/src/ggml-hexagon/htp/main.c#L1010-L1200)

이것이 **DSP 측 Op 디스패치의 핵심 진입점**:

```c
static void htp_packet_callback(dspqueue_t queue, int error, void * context) {
    while (1) {
        // 1. 큐에서 패킷 비-블로킹 읽기
        dspqueue_read_noblock(queue, &flags, 
                              HTP_MAX_PACKET_BUFFERS, &n_bufs, bufs,
                              sizeof(req), &req_size, &req);

        if (err == AEE_EWOULDBLOCK) return; // 모든 패킷 소비 완료

        // 2. Early wakeup 처리
        if (req.flags & HTP_OPFLAGS_EARLY_WAKEUP)
            dspqueue_write_early_wakeup_noblock(ctx->queue, 10, 0);

        // 3. Op 별 디스패치
        switch (req.op) {
            case HTP_OP_MUL_MAT:      proc_matmul_req(ctx, &req, bufs, n_bufs); break;
            case HTP_OP_MUL_MAT_ID:   proc_matmul_id_req(ctx, &req, bufs, n_bufs); break;
            case HTP_OP_MUL:
            case HTP_OP_ADD:
            case HTP_OP_SUB:
            case HTP_OP_DIV:          proc_binary_req(ctx, &req, bufs); break;
            case HTP_OP_RMS_NORM:
            case HTP_OP_SCALE:        proc_unary_req(ctx, &req, bufs); break;
            case HTP_OP_SQR:
            case HTP_OP_SQRT:         proc_unary_req(ctx, &req, bufs); break;
            case HTP_OP_SUM_ROWS:     proc_sum_rows_req(ctx, &req, bufs); break;
            case HTP_OP_UNARY_SILU:
            case HTP_OP_UNARY_GELU:   proc_activations_req(ctx, &req, bufs, n_bufs); break;
            case HTP_OP_GLU_SWIGLU:
            case HTP_OP_GLU_SWIGLU_OAI:
            case HTP_OP_SOFTMAX:
            case HTP_OP_GLU_GEGLU:    proc_activations_req(ctx, &req, bufs, n_bufs); break;
            case HTP_OP_ADD_ID:       proc_add_id_req(ctx, &req, bufs); break;
            case HTP_OP_ROPE:         proc_rope_req(ctx, &req, bufs, n_bufs); break;
            case HTP_OP_FLASH_ATTN_EXT: proc_flash_attn_ext_req(ctx, &req, bufs, n_bufs); break;
            case HTP_OP_SET_ROWS:     proc_set_rows_req(ctx, &req, bufs); break;
            case HTP_OP_GET_ROWS:     proc_get_rows_req(ctx, &req, bufs); break;
            case HTP_OP_CPY:          proc_cpy_req(ctx, &req, bufs); break;
            case HTP_OP_ARGSORT:      proc_argsort_req(ctx, &req, bufs); break;
            case HTP_OP_SSM_CONV:     proc_ssm_conv_req(ctx, &req, bufs); break;
        }
    }
}
```

---

## 4. `proc_*_req()` 패턴 (모든 핸들러가 동일한 패턴)

예시: `proc_matmul_req()` — [main.c L395~L430](../../ggml/src/ggml-hexagon/htp/main.c#L395-L430)

```c
static void proc_matmul_req(struct htp_context *ctx, struct htp_general_req *req,
                            struct dspqueue_buffer *bufs, size_t n_bufs) {
    // 1. 응답 버퍼 설정 (출력 버퍼의 캐시 관리 플래그)
    struct dspqueue_buffer rsp_bufs[1];
    rsp_bufs[0] = bufs[2];  // dst 버퍼
    rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                        DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    // 2. htp_ops_context 구성
    struct htp_ops_context octx = { 0 };
    octx.ctx  = ctx;
    octx.src0 = req->src0;
    octx.src1 = req->src1;
    octx.dst  = req->dst;
    octx.flags = req->flags;
    octx.op    = req->op;

    // 3. 데이터 포인터 업데이트 (버퍼 fd → 실제 DSP 주소)
    octx.src0.data = (uint32_t) bufs[0].ptr;
    octx.src1.data = (uint32_t) bufs[1].ptr;
    octx.dst.data  = (uint32_t) bufs[2].ptr;
    octx.n_threads = ctx->n_threads;

    // 4. 프로파일링 시작
    struct profile_data prof;
    profile_start(&prof);

    // 5. VTCM 획득 → 커널 실행 → VTCM 해제
    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        rsp_status = op_matmul(&octx);  // ← HVX 커널 호출
        vtcm_release(ctx);
    }

    // 6. 프로파일링 종료 + 응답 전송
    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, 1, &prof);
}
```

### 핵심 패턴 요약:
1. 응답 버퍼 캐시 플래그 설정
2. `htp_ops_context` 구성 (req → octx 매핑)
3. `bufs[N].ptr`를 통해 DSP 측 실제 주소로 변환
4. `vtcm_acquire()` → `op_*()`  → `vtcm_release()` (항상 이 패턴)
5. `send_htp_rsp()`로 결과 + 프로파일링 데이터 반환

---

## 5. VTCM 관리

**위치**: [main.c L140~L195](../../ggml/src/ggml-hexagon/htp/main.c#L140-L195)

- 기본 **8MB VTCM** 할당 (`vtcm_alloc`)
- `HAP_compute_res_acquire_cached` / `HAP_compute_res_release_cached` 사용
- **HMX 리소스도 동시 할당**: `HAP_compute_res_attr_set_hmx_param(&attr, 1)` — [main.c L213](../../ggml/src/ggml-hexagon/htp/main.c#L213)
- 세션 간 VTCM 공유 메커니즘: `vtcm_release_callback`으로 우선순위 기반 양보

> **HMX Op 추가 시**: VTCM + HMX가 이미 함께 할당되므로 별도 리소스 획득 불필요.

---

## 6. Op 커널 함수 선언

**위치**: [htp/htp-ops.h](../../ggml/src/ggml-hexagon/htp/htp-ops.h)

```c
int op_matmul(struct htp_ops_context * octx);
int op_matmul_id(struct htp_ops_context * octx);
int op_binary(struct htp_ops_context * octx);
int op_unary(struct htp_ops_context * octx);
int op_sum_rows(struct htp_ops_context * octx);
int op_activations(struct htp_ops_context * octx);
int op_softmax(struct htp_ops_context * octx);
int op_add_id(struct htp_ops_context * octx);
int op_rope(struct htp_ops_context * octx);
int op_flash_attn_ext(struct htp_ops_context * octx);
int op_set_rows(struct htp_ops_context * octx);
int op_get_rows(struct htp_ops_context * octx);
int op_cpy(struct htp_ops_context * octx);
int op_argsort(struct htp_ops_context * octx);
int op_ssm_conv(struct htp_ops_context * octx);
```

### Op 커널 소스 파일:

| 파일 | 구현 Op | 라인 수 |
|------|---------|---------|
| [matmul-ops.c](../../ggml/src/ggml-hexagon/htp/matmul-ops.c) | `op_matmul`, `op_matmul_id` | 2572 |
| [binary-ops.c](../../ggml/src/ggml-hexagon/htp/binary-ops.c) | `op_binary` | - |
| [unary-ops.c](../../ggml/src/ggml-hexagon/htp/unary-ops.c) | `op_unary` | - |
| [act-ops.c](../../ggml/src/ggml-hexagon/htp/act-ops.c) | `op_activations` | - |
| [softmax-ops.c](../../ggml/src/ggml-hexagon/htp/softmax-ops.c) | `op_softmax` | - |
| [rope-ops.c](../../ggml/src/ggml-hexagon/htp/rope-ops.c) | `op_rope` | - |
| [flash-attn-ops.c](../../ggml/src/ggml-hexagon/htp/flash-attn-ops.c) | `op_flash_attn_ext` | 714 |
| [set-rows-ops.c](../../ggml/src/ggml-hexagon/htp/set-rows-ops.c) | `op_set_rows` | - |
| [get-rows-ops.c](../../ggml/src/ggml-hexagon/htp/get-rows-ops.c) | `op_get_rows` | - |
| [cpy-ops.c](../../ggml/src/ggml-hexagon/htp/cpy-ops.c) | `op_cpy` | - |
| [argsort-ops.c](../../ggml/src/ggml-hexagon/htp/argsort-ops.c) | `op_argsort` | - |
| [sum-rows-ops.c](../../ggml/src/ggml-hexagon/htp/sum-rows-ops.c) | `op_sum_rows` | - |
| [ssm-conv.c](../../ggml/src/ggml-hexagon/htp/ssm-conv.c) | `op_ssm_conv` | - |

---

## 7. Worker Pool (멀티스레드 실행)

**위치**: [htp/worker-pool.h](../../ggml/src/ggml-hexagon/htp/worker-pool.h)

```c
typedef void (*worker_callback_t)(unsigned int n, unsigned int i, void *);
```

- 최대 10개 HVX 하드웨어 스레드 (`HTP_MAX_NTHREADS = 10`)
- `worker_pool_run_func(ctx, func, data, n)`: func(n, i, data)를 n개 스레드로 병렬 실행
- 각 op_* 커널이 내부적으로 `worker_pool_run_func`을 사용하여 행 단위 병렬화

---

## 8. DMA 관리

**위치**: [htp/hex-dma.h](../../ggml/src/ggml-hexagon/htp/hex-dma.h), [htp/hex-dma.c](../../ggml/src/ggml-hexagon/htp/hex-dma.c)

- Hexagon UDMA (type1 descriptor) 기반
- DDR ↔ VTCM 간 데이터 이동에 사용
- `dma_queue_push()`: 비동기 DMA 전송 큐잉
- `dma_queue_flush()`: 대기 중인 DMA 완료까지 대기
- `dmstart()`, `dmlink()`, `dmwait()`: 하드웨어 DMA 제어 인라인 함수

---

## 9. HMX Op 추가 시 DSP 측 작업 요약

### 필수 수정 사항:

1. **`htp-msg.h`**: `enum htp_op`에 새 Op 코드 추가
2. **`htp-ops.h`**: 새 커널 함수 선언 추가 (예: `int op_hmx_matmul(struct htp_ops_context * octx);`)
3. **새 커널 파일 생성**: 예: `hmx-matmul-ops.c`
   - `htp_ops_context`에서 텐서 정보 추출
   - VTCM에 DMA로 데이터 로드
   - HMX 커널 호출 (`htp-ops-lib`의 함수들)
   - 결과 DMA로 DDR에 쓰기
   - `HTP_STATUS_OK` 반환
4. **`main.c`**: `htp_packet_callback()`의 `switch`에 `case HTP_OP_HMX_*:` 추가 및 `proc_hmx_*_req()` 핸들러 작성
5. **`htp/CMakeLists.txt`**: 새 소스 파일 추가

### 기존 인프라 재사용 가능 항목:
- `worker_pool_run_func()` — 멀티스레드
- `dma_queue_push()` / `dma_queue_flush()` — DMA 전송
- `vtcm_acquire()` / `vtcm_release()` — VTCM 관리
- `profile_start()` / `profile_stop()` — 프로파일링
- `send_htp_rsp()` — 응답 전송

