# ggml-hexagon vs htp-ops-lib DSP 리소스 관리 비교 분석

> 작성일: 2026-03-11  
> 목적: 두 시스템이 동일 DSP에서 동시 실행 시 리소스 충돌 가능성 분석

---

## 1. DSP 세션 (FastRPC / dspqueue)

### ggml-hexagon 관리 방식

**호스트 측** ([ggml/src/ggml-hexagon/ggml-hexagon.cpp](ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1619-L1674)):
1. `remote_session_control(FASTRPC_GET_URI, ...)` 로 멀티 세션 URI 획득 (L1586-L1596)
2. `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)` 로 Unsigned PD 활성화 (L1607-L1616)
3. `htp_iface_open(session_uri, &handle)` — FastRPC를 통해 DSP 쪽 `remote_handle64` 오픈 (L1619)
4. `dspqueue_create()` — CPU 측에서 dspqueue 생성 (128KB req / 64KB resp) (L1642-L1650)
5. `dspqueue_export()` — queue ID 내보내기 (L1658)
6. `htp_iface_start(handle, dev_id, queue_id, n_hvx)` — DSP 쪽에서 queue import + 리소스 초기화 (L1674)

**DSP 측** ([ggml/src/ggml-hexagon/htp/main.c](ggml/src/ggml-hexagon/htp/main.c#L29-L107)):
- `htp_iface_open()`: `htp_context` 할당, `HAP_power_set()` 으로 전력 설정 (L29-L104)
- `htp_iface_start()`: `dspqueue_import()` 로 queue 가져오기, VTCM 할당, worker pool 초기화 (L253-L307)
- `htp_iface_stop()`: `dspqueue_close()`, worker pool 해제, VTCM 해제 (L309-L340)
- `htp_iface_close()`: context free (L106-L117)

**IDL 인터페이스** ([ggml/src/ggml-hexagon/htp/htp_iface.idl](ggml/src/ggml-hexagon/htp/htp_iface.idl)):
```idl
interface htp_iface : remote_handle64 {
    AEEResult start(in uint32 sess_id, in uint64 dsp_queue_id, in uint32 n_hvx);
    AEEResult stop();
    AEEResult enable_etm();
    AEEResult disable_etm();
};
```

### htp-ops-lib 관리 방식

**호스트 측** ([htp-ops-lib/src/host/session.c](htp-ops-lib/src/host/session.c#L15-L97)):
1. `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)` (L32-L45)
2. `htp_ops_open(uri_domain, &session_handle)` — 글로벌 단일 핸들 (L67)
3. `remote_handle64_control(DSPRPC_CONTROL_LATENCY, ...)` — QoS 모드 (L80-L84)
4. `htp_ops_init_backend()` — 백엔드 초기화: power, vtcm, hmx 순서 (L92)
5. `htp_ops_create_channel()` — 공유 메모리 기반 메시지 채널 (dspqueue 아님) (L96)

**DSP 측** ([htp-ops-lib/src/dsp/commu.c](htp-ops-lib/src/dsp/commu.c#L213-L265)):
- `htp_ops_open()`: 더미 핸들, `message_channel_init()` (L215-L220)
- `htp_ops_init_backend()`: `power_setup()` → `vtcm_manager_setup()` → `hmx_manager_setup()` (L243-L250)
- `htp_ops_create_channel()`: `HAP_mmap_get()` + 폴링 스레드 생성 (L253-L258)
- `htp_ops_close()`: `mmap_manager_release_all()`, `message_channel_destroy()`, HMX/VTCM/Power reset (L223-L232)

**IDL 인터페이스** ([htp-ops-lib/include/htp_ops.idl](htp-ops-lib/include/htp_ops.idl)):
```idl
interface htp_ops : remote_handle64 {
    AEEResult init_backend();
    AEEResult create_channel(in int32 fd, in uint32 size);
    AEEResult destroy_channel();
    AEEResult rms_norm_f32(...);
    AEEResult mat_mul_permuted_w16a32(...);
    AEEResult test_ops();
};
```

### 충돌 분석: DSP 세션

| 항목 | 상세 |
|------|------|
| **통신 방식** | ggml-hexagon: `dspqueue` (HW-assisted 비동기 큐) / htp-ops-lib: 공유메모리 폴링 + FastRPC |
| **세션 격리** | 서로 다른 IDL 인터페이스 (`htp_iface` vs `htp_ops`)를 사용하므로 별도의 `remote_handle64`를 가짐 |
| **충돌 가능성** | **낮음** — FastRPC는 멀티 세션을 지원하며, 서로 다른 skel 라이브러리로 로드됨. 단, 동일 Unsigned PD 내에서 실행될 수 있어 메모리 공간은 공유됨 |

---

## 2. VTCM (Vector Tightly Coupled Memory)

### ggml-hexagon 관리 방식

**할당** ([ggml/src/ggml-hexagon/htp/main.c](ggml/src/ggml-hexagon/htp/main.c#L201-L238)):
```c
static int vtcm_alloc(struct htp_context * ctx) {
    unsigned int vtcm_size = 8 * 1024 * 1024;  // 8MB default
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);  // 실제 가능 크기 쿼리

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);       // 비직렬화 모드
    HAP_compute_res_attr_set_cache_mode(&attr, 1);      // 캐시 모드 활성화
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, 0, vtcm_size);  // 전체 VTCM 요청
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (void *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);       // HMX도 함께 요청

    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000);  // 1초 타임아웃
    // ...
    ctx->vtcm_base = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size = vtcm_size;
    ctx->vtcm_rctx = rctx;
}
```

**사용 패턴** — acquire/release per-Op ([ggml/src/ggml-hexagon/htp/main.c](ggml/src/ggml-hexagon/htp/main.c#L141-L197)):
- Op 실행 전 `vtcm_acquire()` → Op 실행 → `vtcm_release()` (L439-L441 등 모든 proc_*_req 함수)
- `HAP_compute_res_acquire_cached()` / `HAP_compute_res_release_cached()` 사용
- **릴리스 콜백** (`vtcm_release_callback`, L183-L197): 다른 세션이 VTCM을 요청하면 콜백으로 양보 가능
- 스레드 우선순위를 일시적으로 올려서 VTCM 획득 보장 (L150-L165)

**컨텍스트 저장** ([ggml/src/ggml-hexagon/htp/htp-ctx.h](ggml/src/ggml-hexagon/htp/htp-ctx.h#L18-L30)):
```c
uint8_t * vtcm_base;           // VTCM 베이스 주소
size_t    vtcm_size;           // VTCM 크기
uint32_t  vtcm_rctx;           // compute_res 컨텍스트 ID
atomic_bool vtcm_valid;        // VTCM 현재 유효 여부
atomic_bool vtcm_inuse;        // VTCM 현재 사용 중 여부
atomic_bool vtcm_needs_release; // 릴리스 보류 플래그
```

### htp-ops-lib 관리 방식

**할당** ([htp-ops-lib/src/dsp/vtcm_mgr.cc](htp-ops-lib/src/dsp/vtcm_mgr.cc#L24-L53)):
```cpp
void vtcm_manager_setup() {
    HAP_compute_res_query_VTCM(0, &total_size, &total_pages, &avail_size, &avail_pages);

    compute_res_attr_t req;
    HAP_compute_res_attr_init(&req);
    HAP_compute_res_attr_set_vtcm_param(&req, total_size, 1);  // 전체 VTCM, 단일 페이지

    vtcm_mgr_ctx_id = HAP_compute_res_acquire(&req, 10000);    // 10ms 타임아웃
    vtcm_base = (uint8_t *) HAP_compute_res_attr_get_vtcm_ptr(&req);
    memset(vtcm_base, 0, total_size);
    vtcm_reserved_start = vtcm_base + total_size;
}
```

**사용 패턴**:
- **초기화 시 전체 VTCM을 획득하고 세션 종료까지 계속 점유** (L47: `HAP_compute_res_acquire`)
- `vtcm_manager_reserve_area()` 로 명명된 영역을 끝에서부터 할당 ([vtcm_mgr.cc](htp-ops-lib/src/dsp/vtcm_mgr.cc#L67-L82))
- `vtcm_seq_alloc()` 으로 순차 할당 (bump allocator)
- **릴리스 콜백 없음** — VTCM을 양보하는 메커니즘 없음
- `vtcm_manager_reset()` 시에만 해제 ([vtcm_mgr.cc](htp-ops-lib/src/dsp/vtcm_mgr.cc#L55-L59))

### 충돌 분석: VTCM

| 항목 | 상세 |
|------|------|
| **핵심 문제** | **높은 충돌 가능성** |
| **이유** | 두 시스템 모두 **전체 VTCM을 요청**함. ggml-hexagon은 `HAP_compute_res_query_VTCM`으로 쿼리한 전체 크기를, htp-ops-lib도 `total_size`를 요청 |
| **ggml-hexagon 특징** | 릴리스 콜백 지원 (`vtcm_release_callback`), cached acquire/release로 Op 단위 양보 가능, 우선순위 범프로 경쟁 해결 |
| **htp-ops-lib 특징** | 릴리스 콜백 없음, 10ms 짧은 타임아웃으로 획득 시도, 세션 전체 기간 동안 점유 |
| **동시 실행 시** | htp-ops-lib가 먼저 VTCM을 잡으면 ggml-hexagon은 릴리스 콜백을 받을 수 없어 1초 타임아웃 후 `abort()` 호출. 반대로 ggml-hexagon이 먼저 잡으면 htp-ops-lib는 10ms 타임아웃에 실패 |
| **해결 방안** | (1) VTCM 분할 할당, (2) htp-ops-lib에 릴리스 콜백 추가, (3) 한쪽만 VTCM 사용하도록 직렬화 |

---

## 3. HVX 스레드 / Worker Pool

### ggml-hexagon 관리 방식

**초기화** ([ggml/src/ggml-hexagon/htp/main.c](ggml/src/ggml-hexagon/htp/main.c#L281-L304)):
```c
// htp_iface_start() 내부
qurt_sysenv_get_max_hw_threads(&hw_threads);
uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;
if (n_hvx == 0) n_hvx = hw_nhvx;         // 기본: 모든 HVX 유닛 사용
if (n_hvx > hw_threads.max_hthreads) n_hvx = hw_threads.max_hthreads;
ctx->n_threads = n_hvx;
worker_pool_init(&ctx->worker_pool, n_hvx);
```

**Worker Pool 구현** ([ggml/src/ggml-hexagon/htp/worker-pool.c](ggml/src/ggml-hexagon/htp/worker-pool.c)):
- `qurt_futex_wait`/`qurt_futex_wake` 기반 동기화 (L50-L66)
- 호출 스레드가 job #0 실행, 나머지를 worker에게 분배 (L207-L222)
- `MAX_NUM_WORKERS = 10` ([worker-pool.h](ggml/src/ggml-hexagon/htp/worker-pool.h#L29))
- 단일 메모리 블록 할당 (스택 + 구조체) (L83-L92)
- 우선순위: 생성 스레드의 우선순위를 상속 (L131-L140)

### htp-ops-lib 관리 방식

**Worker Pool 구현** ([htp-ops-lib/src/dsp/worker_pool.c](htp-ops-lib/src/dsp/worker_pool.c)):
- `MAX_NUM_WORKERS = 6` ([worker_pool.h](htp-ops-lib/include/dsp/worker_pool.h#L80))
- `qurt_anysignal` 기반 신호 동기화 (L101-L120)
- **생성자(`__attribute__((constructor))`)에서 정적 worker pool 자동 생성** (L134-L153)
- `worker_pool_init_ex()`: 커스텀 worker 수, HMX 허용 플래그 지원 (L155-L234)
- HMX worker pool: 별도로 1개 스레드의 전용 HMX 풀 생성 ([hmx_mgr.c](htp-ops-lib/src/dsp/hmx_mgr.c#L27))
- `worker_pool_submit()`: 비동기 작업 제출 + synctoken 기반 대기 (L290-L345)

### 충돌 분석: HVX 스레드

| 항목 | 상세 |
|------|------|
| **핵심 문제** | **중간 충돌 가능성** |
| **이유** | 두 시스템 모두 가용 HW 스레드 수만큼 스레드를 생성함 |
| **ggml-hexagon** | 최대 10개 스레드 (HW 스레드 수까지), 세션에 종속적 (start/stop 생명주기) |
| **htp-ops-lib** | 최대 6개 스레드 + 정적 풀(라이브러리 로드 시 자동 생성) + 1개 HMX 전용 스레드 + 1개 폴링 스레드 |
| **동시 실행 시** | QuRT는 SW 스레드를 HW 스레드에 스케줄링하므로 스레드 생성 자체는 실패하지 않음. 그러나 총 스레드 수가 HW 스레드 수(보통 4~6개)를 초과하면 컨텍스트 스위칭 오버헤드로 성능 저하 발생 |
| **HVX 유닛 경쟁** | HVX `lock`/`unlock`은 QuRT가 시분할 관리하므로 크래시는 아니지만 성능 저하 가능 |

---

## 4. HMX (Hexagon Matrix eXtension)

### ggml-hexagon 관리 방식

**전원** ([ggml/src/ggml-hexagon/htp/main.c](ggml/src/ggml-hexagon/htp/main.c#L91-L101)):
```c
// htp_iface_open() 내부
HAP_power_request_t request;
request.type         = HAP_power_set_HMX;
request.hmx.power_up = TRUE;
HAP_power_set((void *) &ctx, &request);
```

**리소스** ([ggml/src/ggml-hexagon/htp/main.c](ggml/src/ggml-hexagon/htp/main.c#L210)):
```c
// vtcm_alloc() 내부에서 HMX 리소스도 함께 요청
HAP_compute_res_attr_set_hmx_param(&attr, 1);
```
- VTCM compute_res 컨텍스트에 HMX를 **번들로 함께 획득**
- 별도의 HMX lock/unlock API 호출 없음 — compute_res 수준에서 관리

### htp-ops-lib 관리 방식

**전원** ([htp-ops-lib/src/dsp/power.c](htp-ops-lib/src/dsp/power.c#L39-L46)):
```c
// power_setup() 내부
req.type         = HAP_power_set_HMX;
req.hmx.power_up = TRUE;
HAP_power_set(&power_ctx, &req);
```

**리소스** ([htp-ops-lib/src/dsp/hmx_mgr.c](htp-ops-lib/src/dsp/hmx_mgr.c#L12-L28)):
```c
void hmx_manager_setup() {
    compute_res_attr_t req;
    HAP_compute_res_attr_init(&req);
    HAP_compute_res_attr_set_hmx_param(&req, 1);
    hmx_mgr_ctx_id = HAP_compute_res_acquire(&req, 10000);  // 별도의 HMX 전용 ctx

    worker_pool_init_ex(&hmx_worker_pool_ctx, 8192, 1, 1);  // HMX 전용 1-스레드 풀
}
```

**스레드 수준 Lock** ([htp-ops-lib/src/dsp/hmx_mgr.c](htp-ops-lib/src/dsp/hmx_mgr.c#L42-L57)):
```c
void hmx_manager_enable_execution() {
    HAP_compute_res_hmx_lock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
}
void hmx_manager_disable_execution() {
    HAP_compute_res_hmx_unlock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
}
```

**유닛 수준 스핀락** ([htp-ops-lib/src/dsp/hmx_mgr.c](htp-ops-lib/src/dsp/hmx_mgr.c#L59-L75)):
```c
void hmx_unit_acquire() {
    // 어셈블리 기반 memw_locked 스핀락
}
void hmx_unit_release() {
    *(volatile int *) &hmx_mgr_spin_lock = 0;
}
```

### 충돌 분석: HMX

| 항목 | 상세 |
|------|------|
| **핵심 문제** | **중간~높은 충돌 가능성** |
| **ggml-hexagon** | HMX를 VTCM compute_res와 번들로 획득. VTCM acquire 시 HMX도 함께 잡힘 |
| **htp-ops-lib** | HMX를 별도 `HAP_compute_res_acquire`로 획득 + `HAP_compute_res_hmx_lock2` shared 모드 사용 + 자체 스핀락 |
| **동시 실행 시** | ggml-hexagon이 VTCM+HMX를 번들로 잡고 있을 때, htp-ops-lib의 별도 HMX acquire가 성공할 수 있으나 실제 HMX 실행이 직렬화될 수 있음. shared lock 모드이므로 크래시보다는 성능 저하 가능 |

---

## 5. 전력 관리 (Power / DCVS)

### ggml-hexagon 관리 방식

([ggml/src/ggml-hexagon/htp/main.c](ggml/src/ggml-hexagon/htp/main.c#L44-L89)):
```c
// 1) 클라이언트 클래스 설정
request.type    = HAP_power_set_apptype;
request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

// 2) DCVS v3 설정
request.dcvs_v3.dcvs_option               = HAP_DCVS_V2_PERFORMANCE_MODE;
request.dcvs_v3.bus_params.min_corner     = HAP_DCVS_VCORNER_MAX;    // 최대 전력
request.dcvs_v3.bus_params.max_corner     = HAP_DCVS_VCORNER_MAX;
request.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_MAX;
request.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_MAX;
request.dcvs_v3.sleep_disable             = TRUE;                     // 슬립 비활성화

// 3) HVX 전원
request.type         = HAP_power_set_HVX;
request.hvx.power_up = TRUE;

// 4) HMX 전원
request.type         = HAP_power_set_HMX;
request.hmx.power_up = TRUE;
```
- **voltage corner**: `HAP_DCVS_VCORNER_MAX` (최대 성능)
- **슬립**: 비활성화
- **컨텍스트**: `(void *) ctx` — 세션별 핸들 사용

### htp-ops-lib 관리 방식

([htp-ops-lib/src/dsp/power.c](htp-ops-lib/src/dsp/power.c#L10-L56)):
```c
// DCVS v3 설정
req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
req.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_NOM;        // 노미널
req.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_TURBO_L3;   // 터보 L3
req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;
req.dcvs_v3.bus_params.min_corner     = HAP_DCVS_VCORNER_NOM;
req.dcvs_v3.bus_params.max_corner     = HAP_DCVS_VCORNER_TURBO_L3;
req.dcvs_v3.latency                   = 100;  // 100us 레이턴시 타겟

// HMX 전원
req.type         = HAP_power_set_HMX;
req.hmx.power_up = TRUE;
```
- **voltage corner**: `HAP_DCVS_VCORNER_NOM` ~ `HAP_DCVS_VCORNER_TURBO_L3` (노미널~터보)
- **컨텍스트**: `&power_ctx` — 정적 변수 사용
- **HVX 전원 명시적 설정 없음** (HVX는 일반적으로 기본 활성화)
- `power_reset()` 에서 HMX 전원 끄기 + DCVS 초기화 (L49-L56)

### 충돌 분석: 전력

| 항목 | 상세 |
|------|------|
| **핵심 문제** | **낮은 충돌 가능성** |
| **이유** | `HAP_power_set()`은 투표(voting) 방식으로 동작. 복수의 클라이언트가 전력 요청을 보내면 가장 높은 요청이 적용됨 |
| **차이점** | ggml-hexagon은 `VCORNER_MAX` (최대), htp-ops-lib는 `VCORNER_TURBO_L3` (터보). 동시 실행 시 MAX가 적용됨 |
| **부작용** | htp-ops-lib의 전력 요청은 ggml-hexagon보다 보수적이므로, 동시 실행 시 ggml-hexagon의 MAX 설정이 htp-ops-lib에도 혜택을 줌 |
| **주의점** | htp-ops-lib의 `power_reset()` 호출 시 DCVS 설정 초기화 + HMX 전원 off → ggml-hexagon이 실행 중이면 간접 영향 가능 (하지만 HAP_power_set은 클라이언트별로 관리되므로 일반적으로 안전) |

---

## 6. 종합 충돌 위험 매트릭스

| 리소스 | 충돌 위험 | 심각도 | 설명 |
|--------|----------|--------|------|
| **DSP 세션** | 🟢 낮음 | 낮음 | 별도의 FastRPC 인터페이스/핸들 사용 |
| **VTCM** | 🔴 **높음** | **치명적** | 두 시스템 모두 전체 VTCM 요청. 한쪽이 잡으면 다른 쪽 실패/abort |
| **HVX 스레드** | 🟡 중간 | 성능 저하 | HW 스레드 초과 시 컨텍스트 스위칭 오버헤드 |
| **HMX** | 🟡 중간~높음 | 성능 저하~장애 | VTCM 번들 vs 별도 acquire. compute_res 경합 가능 |
| **전력** | 🟢 낮음 | 무시 | 투표 기반, 더 높은 요청이 적용 |

---

## 7. 권장 사항

### 즉시 해결 (동시 실행 불가 원인 제거)

1. **VTCM 분할 또는 시분할**
   - 옵션 A: VTCM을 두 시스템에 정적 분할 (예: 각 4MB)
   - 옵션 B: htp-ops-lib에 릴리스 콜백 추가하여 ggml-hexagon과 시분할 공유
   - 옵션 C: 하나의 VTCM 매니저로 통합

2. **HMX 리소스 통합**
   - ggml-hexagon의 VTCM+HMX 번들 acquire를 풀고, 공유 HMX 매니저 사용 검토

### 중기 최적화

3. **Worker Pool 통합**
   - 하나의 worker pool을 공유하여 HW 스레드 과점유 방지
   - 또는 각 시스템의 스레드 수를 총 HW 스레드 수의 절반으로 제한

4. **전력 정책 통합**
   - 공통 전력 매니저를 두어 일관된 DCVS 정책 적용

### 장기 아키텍처

5. **단일 DSP 백엔드로 통합**
   - htp-ops-lib의 op 구현을 ggml-hexagon의 dspqueue 프레임워크로 마이그레이션
   - 리소스 관리를 단일 지점에서 수행
