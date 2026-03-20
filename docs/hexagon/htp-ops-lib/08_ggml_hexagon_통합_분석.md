# htp-ops-lib ↔ llama.cpp Hexagon 백엔드 통합 분석

> 분석 일시: 2026-03-11  
> 분석 대상: `ggml/src/ggml-hexagon/` (llama.cpp 공식 Hexagon 백엔드) 및 `htp-ops-lib/` (연구 프로토타입 커스텀 Op 라이브러리)

---

## 목차

1. [두 시스템 개요 비교](#1-두-시스템-개요-비교)
2. [llama.cpp Hexagon 백엔드 상세 분석](#2-llamacpp-hexagon-백엔드-상세-분석)
3. [htp-ops-lib 상세 분석](#3-htp-ops-lib-상세-분석)
4. [공유 라이브러리 로딩 메커니즘](#4-공유-라이브러리-로딩-메커니즘)
5. [RPC / 통신 구조 비교](#5-rpc--통신-구조-비교)
6. [Op 디스패치 방식 비교](#6-op-디스패치-방식-비교)
7. [공유 데이터 구조](#7-공유-데이터-구조)
8. [통합 가능성 및 차이점 요약](#8-통합-가능성-및-차이점-요약)

---

## 1. 두 시스템 개요 비교

### 1.1 아키텍처 비교표

| 항목 | llama.cpp Hexagon 백엔드 (`ggml-hexagon`) | htp-ops-lib |
|------|------|------|
| **위치** | `ggml/src/ggml-hexagon/` | `htp-ops-lib/` |
| **목적** | ggml 백엔드 추상화에 맞춘 완전한 Hexagon 백엔드 | 독립적인 연구 프로토타입 Op 라이브러리 |
| **통신 방식** | **dspqueue** (비동기 메시지 큐) | **FastRPC** (IDL 기반) + **공유 메모리 폴링** |
| **IDL 인터페이스** | `htp_iface.idl` (세션 관리만) | `htp_ops.idl` (세션 + Op RPC) |
| **DSP 라이브러리** | `libggml-htp-v{arch}.so` (아키텍처별 skel) | `libhtp_ops_skel.so` |
| **Host 라이브러리** | `libggml-hexagon.so` + `htp_iface_stub.c` | `libhtp_ops.so` (stub) |
| **지원 Op 수** | ~25개 (matmul, rope, flash_attn 등) | 6개 (rms_norm, matmul 변형, flash_attn) |
| **HMX 지원** | 일부 (v73+ 포함) | 핵심적으로 사용 (FP16 HMX 필수) |
| **가중치 Repack** | q4x4x2, q8x4x2, mxfp4x4x2 (CPU측) | Crouton 레이아웃 (DSP측 repacked weights) |

### 1.2 관계 요약

```
                      llama.cpp 공식 백엔드                          연구 프로토타입
                  ┌─────────────────────────────┐              ┌──────────────────────┐
                  │ ggml/src/ggml-hexagon/       │              │ htp-ops-lib/         │
                  │                              │              │                      │
  Host (CPU)      │ ggml-hexagon.cpp             │              │ src/host/session.c   │
                  │ htp-drv.cpp (libcdsprpc 로딩)│              │ src/host/op_export.c │
                  │ htp_iface_stub.c (생성코드)  │              │ htp_ops_stub.c (생성)│
                  │                              │              │                      │
  ────────────── │ ─── dspqueue ─── FastRPC ─── │ ──────────── │ ─── FastRPC ──────── │
                  │                              │              │                      │
  DSP (Hexagon)   │ htp/main.c                   │              │ src/dsp/commu.c      │
                  │ htp/matmul-ops.c             │              │ src/dsp/op_executor.cc│
                  │ htp/flash-attn-ops.c         │              │ src/dsp/ops/mat_mul.c│
                  │ htp/softmax-ops.c  ...       │              │ src/dsp/ops/flash... │
                  └─────────────────────────────┘              └──────────────────────┘
```

**핵심 차이**: `ggml-hexagon`은 **dspqueue 기반 비동기 메시지 패싱**을 사용하고, `htp-ops-lib`는 **FastRPC IDL RPC 호출 + 공유 메모리 폴링 기반 채널**을 사용합니다.

---

## 2. llama.cpp Hexagon 백엔드 상세 분석

### 2.1 핵심 파일 구조

분석 위치: `ggml/src/ggml-hexagon/`

```
ggml/src/ggml-hexagon/
├── ggml-hexagon.cpp          # 메인 백엔드 구현 (3301줄) - 세션, 버퍼, 디스패치, 그래프 최적화
├── htp-drv.cpp               # FastRPC 드라이버 로더 (419줄) - libcdsprpc.dll/.so 동적 로딩
├── htp-drv.h                 # 드라이버 API 선언 (rpcmem, dspqueue, remote_handle 등)
├── libdl.h                   # 크로스 플랫폼 dlopen/LoadLibrary 추상화
├── op-desc.h                 # Op 디버깅용 텐서 디스크립터 포맷터
├── libggml-htp.inf           # Windows 드라이버 INF (skel 배포용)
├── CMakeLists.txt            # 빌드 설정 (host stub + DSP skel 다중 아키텍처 빌드)
└── htp/                      # DSP측 코드 (Hexagon v68~v81용)
    ├── main.c                # DSP 진입점 - dspqueue 콜백 기반 Op 디스패치 (1200줄)
    ├── htp-msg.h             # Host↔DSP 메시지 프로토콜 (req/rsp 구조체)
    ├── htp-ops.h             # DSP Op 인터페이스 선언
    ├── htp-ctx.h             # DSP 컨텍스트 (큐, VTCM, DMA, 워커풀)
    ├── htp_iface.idl         # FastRPC IDL (start/stop/etm만)
    ├── matmul-ops.c          # 행렬곱 구현 (HVX + DMA 타일링)
    ├── flash-attn-ops.c      # Flash Attention 구현
    ├── softmax-ops.c         # Softmax
    ├── rope-ops.c            # RoPE
    ├── binary-ops.c          # 이항 연산 (add, mul, sub, div)
    ├── unary-ops.c           # 단항 연산 (rms_norm, scale, silu, gelu 등)
    ├── cpy-ops.c             # 텐서 복사
    ├── set-rows-ops.c        # Set rows
    ├── get-rows-ops.c        # Get rows
    ├── argsort-ops.c         # Argsort
    ├── sum-rows-ops.c        # Sum rows
    ├── act-ops.c             # Activation 함수
    ├── ssm-conv.c            # SSM Convolution (Mamba)
    ├── worker-pool.c/.h      # HVX 멀티스레드 워커 풀
    ├── hex-dma.c/.h          # DMA 유틸리티
    └── hvx-*.h               # HVX SIMD 유틸리티 (exp, sqrt, sigmoid 등)
```

### 2.2 Op 디스패치 파이프라인

분석 위치: [ggml-hexagon.cpp](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)

#### Step 1: 드라이버 초기화 (`htpdrv_init()`)

```cpp
// 위치: htp-drv.cpp:327-388
int htpdrv_init() {
    // Windows: libcdsprpc.dll을 드라이버 경로에서 로드
    // Linux/Android: libcdsprpc.so를 dlopen
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
    dl_handle_ptr handle { dl_load_library(path) };

    // 함수 포인터 바인딩 (20+ 심볼)
    dlsym(handle, rpcmem_alloc_pfn_t,  rpcmem_alloc_pfn,  rpcmem_alloc,  false);
    dlsym(handle, rpcmem_free_pfn_t,   rpcmem_free_pfn,   rpcmem_free,   false);
    dlsym(handle, rpcmem_to_fd_pfn_t,  rpcmem_to_fd_pfn,  rpcmem_to_fd,  false);
    dlsym(handle, fastrpc_mmap_pfn_t,  fastrpc_mmap_pfn,  fastrpc_mmap,  false);
    dlsym(handle, dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, false);
    dlsym(handle, dspqueue_write_pfn_t,  dspqueue_write_pfn,  dspqueue_write,  false);
    // ... remote_handle64_open, remote_session_control 등
}
```

**동적 로딩 대상**: `libcdsprpc.dll` (Windows) 또는 `libcdsprpc.so` (Android/Linux)에서 **rpcmem, fastrpc, dspqueue, remote_handle** 심볼을 런타임에 바인딩합니다.

#### Step 2: 세션 생성 및 DSP 연결

```cpp
// 위치: ggml-hexagon.cpp:1535-1685
void ggml_hexagon_session::allocate(int dev_id) {
    // 1. CDSP 도메인 획득
    domain * my_domain = get_domain(3);  // CDSP

    // 2. 새 FastRPC 세션 예약 (다중 디바이스 지원)
    remote_session_control(FASTRPC_RESERVE_NEW_SESSION, &n, sizeof(n));

    // 3. DSP skel 열기 (아키텍처별)
    //    uri = "file:///libggml-htp-v73.so?htp_iface_skel_handle_invoke"
    htp_iface_open(session_uri, &this->handle);

    // 4. dspqueue 생성 (128KB req / 64KB rsp)
    dspqueue_create(domain_id, 0, 128*1024, 64*1024, nullptr, nullptr, this, &queue);

    // 5. 큐 ID 내보내기 → DSP에 전달
    dspqueue_export(queue, &queue_id);

    // 6. DSP 서비스 시작
    htp_iface_start(handle, dev_id, queue_id, opt_nhvx);
}
```

#### Step 3: Op 실행 (그래프 컴퓨트)

```cpp
// 위치: ggml-hexagon.cpp:2569
static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_hexagon_dispatch_op<init_binary_req<true>>(sess, node, flags);
                break;
            case GGML_OP_RMS_NORM:
                ggml_hexagon_dispatch_op<init_unary_req>(sess, node, flags);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                ggml_hexagon_dispatch_op<init_flash_attn_ext_req>(sess, node, flags);
                break;
            // ... 총 25가지 Op
        }
    }
    sess->flush();  // 모든 pending Op 완료 대기
}
```

#### Step 4: dspqueue 기반 Op 전송/수신

```cpp
// 위치: ggml-hexagon.cpp:2265-2291
template <htp_req_init_func_t _init_req_func>
static void ggml_hexagon_dispatch_op(ggml_hexagon_session *sess, const ggml_tensor * op, uint32_t flags) {
    htp_general_req req;
    memset(&req, 0, sizeof(req));
    req.flags = flags;

    dspqueue_buffer bufs[HTP_MAX_PACKET_BUFFERS];
    size_t n_bufs = _init_req_func(&req, bufs, op);

    sess->enqueue(req, bufs, n_bufs, opt_opsync);
}

// enqueue → dspqueue_write
void ggml_hexagon_session::enqueue(struct htp_general_req &req, ...) {
    this->op_pending++;
    dspqueue_write(this->queue, 0, n_bufs, bufs, sizeof(req), (const uint8_t *)&req, DSPQUEUE_TIMEOUT);
}

// flush → dspqueue_read 반복
void ggml_hexagon_session::flush() {
    while (this->op_pending) {
        dspqueue_read(q, &flags, ...);     // 응답 수신
        this->op_pending--;
    }
}
```

### 2.3 DSP측 패킷 처리

분석 위치: [htp/main.c](../../ggml/src/ggml-hexagon/htp/main.c)

```c
// 위치: htp/main.c:1038-1195
static void htp_packet_callback(dspqueue_t queue, int error, void *context) {
    while (1) {
        // 비동기적으로 큐에서 패킷 읽기
        dspqueue_read_noblock(queue, &flags, HTP_MAX_PACKET_BUFFERS, &n_bufs, bufs,
                              sizeof(req), &req_size, (uint8_t *)&req);

        // Op 종류에 따라 분기
        switch (req.op) {
            case HTP_OP_MUL_MAT:    proc_matmul_req(ctx, &req, bufs, n_bufs);    break;
            case HTP_OP_RMS_NORM:   proc_unary_req(ctx, &req, bufs);             break;
            case HTP_OP_SOFTMAX:    proc_activations_req(ctx, &req, bufs, n_bufs); break;
            case HTP_OP_ROPE:       proc_rope_req(ctx, &req, bufs, n_bufs);      break;
            case HTP_OP_FLASH_ATTN_EXT: proc_flash_attn_ext_req(ctx, &req, bufs, n_bufs); break;
            // ... 기타
        }
    }
}
```

각 `proc_*_req()` 함수 내부:
1. `htp_ops_context` 구성 (버퍼 포인터 매핑)
2. **VTCM 획득** (`vtcm_acquire`)
3. **Op 커널 실행** (예: `op_matmul(&octx)`)
4. **VTCM 해제** (`vtcm_release`)
5. **프로파일링 데이터 수집** → 응답 전송 (`send_htp_rsp`)

### 2.4 이미 포함된 FastRPC/rpcmem 참조

`ggml-hexagon` 백엔드는 **이미 전면적으로 FastRPC 인프라를 사용**합니다:

| 심볼/API | 위치 | 용도 |
|----------|------|------|
| `rpcmem_alloc2` | [ggml-hexagon.cpp:278](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L278) | 공유 메모리 버퍼 할당 |
| `rpcmem_to_fd` | [ggml-hexagon.cpp:284](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L284) | FD 변환 |
| `rpcmem_free` | [ggml-hexagon.cpp:296](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L296) | 해제 |
| `fastrpc_mmap` | [ggml-hexagon.cpp:244](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L244) | 버퍼를 DSP 주소 공간에 매핑 |
| `fastrpc_munmap` | [ggml-hexagon.cpp:261](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L261) | 매핑 해제 |
| `dspqueue_write`/`read` | [ggml-hexagon.cpp:146-186](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L146) | Op 요청/응답 (비동기 큐) |
| `remote_handle64_open` | [htp-drv.cpp:170](../../ggml/src/ggml-hexagon/htp-drv.cpp#L170) | DSP 세션 열기 |
| `remote_session_control` | [htp-drv.cpp:186](../../ggml/src/ggml-hexagon/htp-drv.cpp#L186) | 세션 제어 (unsigned PD 등) |
| `remote_handle64_control` | [htp-drv.cpp:178](../../ggml/src/ggml-hexagon/htp-drv.cpp#L178) | 핸들 레벨 제어 (QoS 등) |

**결론**: `ggml-hexagon` 자체가 이미 `rpcmem`, `fastrpc`, `dspqueue`를 **완전히** 참조하고 있으며, 별도의 `htp_ops` 라이브러리에 대한 참조는 **없습니다**.

---

## 3. htp-ops-lib 상세 분석

### 3.1 Host측 API

#### `op_export.h` / `op_export.c`

분석 위치: [htp-ops-lib/include/host/op_export.h](../../../htp-ops-lib/include/host/op_export.h), [htp-ops-lib/src/host/op_export.c](../../../htp-ops-lib/src/host/op_export.c)

```c
// op_export.h - RPC 래퍼 함수 선언
int htp_ops_rpc_rms_norm_f32(int dst_fd, int dst_offset, int src_fd, int src_offset, int ne0, int ne1);
int htp_ops_rpc_mat_mul_permuted_w16a32(int output_fd, int output_offset, int activation_fd,
                                        int activation_offset, int weight_fd, int weight_offset,
                                        int m, int k, int n);
```

```c
// op_export.c - 구현: QAIC 생성 stub를 통해 DSP로 RPC 호출
int htp_ops_rpc_rms_norm_f32(int dst_fd, int dst_offset, int src_fd, int src_offset, int ne0, int ne1) {
    return htp_ops_rms_norm_f32(get_global_handle(), dst_fd, dst_offset, src_fd, src_offset, ne0, ne1);
}

int htp_ops_rpc_mat_mul_permuted_w16a32(...) {
    return htp_ops_mat_mul_permuted_w16a32(get_global_handle(), ...);
}
```

**특징**: 각 Op가 **개별 FastRPC 호출**로 실행됩니다. `htp_ops_rms_norm_f32()`는 QAIC IDL 컴파일러가 `htp_ops.idl`에서 생성한 `htp_ops_stub.c`의 함수입니다.

#### `session.h` / `session.c`

분석 위치: [htp-ops-lib/include/host/session.h](../../../htp-ops-lib/include/host/session.h), [htp-ops-lib/src/host/session.c](../../../htp-ops-lib/src/host/session.c)

```c
// session.h - DSP 세션 관리 API
int open_dsp_session(int domain_id, int unsigned_pd_enabled);
void close_dsp_session();
remote_handle64 get_global_handle();
void init_htp_backend();
int create_htp_message_channel(int fd, unsigned int max_msg_size);
```

```c
// session.c 핵심 로직
int open_dsp_session(int domain_id, int unsigned_pd_enabled) {
    // 1. unsigned PD 설정
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ctrl, sizeof(ctrl));

    // 2. URI 구성: "file:///libhtp_ops_skel.so?htp_ops_skel_handle_invoke&..."
    snprintf(uri_domain, ..., "%s%s", htp_ops_URI, my_domain->uri);

    // 3. FastRPC 세션 열기
    htp_ops_open(uri_domain, &session_handle);

    // 4. FastRPC QoS 모드 (50us 지연 목표)
    struct remote_rpc_control_latency lat_ctrl = { .enable = RPC_PM_QOS, .latency = 50 };
    remote_handle64_control(session_handle, DSPRPC_CONTROL_LATENCY, &lat_ctrl, sizeof(lat_ctrl));
}

void init_htp_backend() {
    htp_ops_init_backend(session_handle);  // DSP측 초기화 트리거
}

int create_htp_message_channel(int fd, unsigned int max_msg_size) {
    return htp_ops_create_channel(session_handle, fd, max_msg_size);  // 공유 메모리 채널 생성
}
```

### 3.2 IDL 인터페이스

분석 위치: [htp-ops-lib/include/htp_ops.idl](../../../htp-ops-lib/include/htp_ops.idl)

```idl
interface htp_ops : remote_handle64 {
    AEEResult init_backend();
    AEEResult create_channel(in int32 fd, in uint32 size);
    AEEResult destroy_channel();

    /* 개별 Op RPC */
    AEEResult rms_norm_f32(in int32 fd0, in int32 offset0, in int32 fd1, in int32 offset1,
                           in int32 ne0, in int32 ne1);
    AEEResult mat_mul_permuted_w16a32(in int32 fd0, in int32 offset0, ...);
    AEEResult test_ops();
};
```

vs llama.cpp `htp_iface.idl`:

```idl
interface htp_iface : remote_handle64 {
    AEEResult start(in uint32 sess_id, in uint64 dsp_queue_id, in uint32 n_hvx);
    AEEResult stop();
    AEEResult enable_etm();
    AEEResult disable_etm();
};
```

**핵심 차이**: `htp_ops.idl`은 **Op별 개별 RPC를** 정의하지만, `htp_iface.idl`은 **세션 제어만** 정의하고 실제 Op 전송은 **dspqueue** 메시지로 수행합니다.

### 3.3 DSP측 통신 (`commu.c`)

분석 위치: [htp-ops-lib/src/dsp/commu.c](../../../htp-ops-lib/src/dsp/commu.c)

`htp-ops-lib`는 **두 가지 통신 모드**를 지원합니다:

1. **FastRPC Mode**: `htp_ops_rms_norm_f32()` 등 IDL 함수 직접 호출
2. **공유 메모리 채널 Mode**: `create_channel()`로 공유 버퍼를 만들고, DSP 폴링 스레드가 메시지를 감시

```c
// 채널 모드의 DSP 폴링 루프
static void msg_receiver_loop(void *param) {
    while (1) {
        // 캐시 무효화 후 공유 메모리 상태 확인
        qurt_mem_cache_clean((qurt_addr_t)msg_hdr, chan->max_msg_size, QURT_MEM_CACHE_INVALIDATE, ...);

        // acquire 시맨틱으로 상태 읽기
        asm volatile("%0 = memd_aq(%1)" : "=r"(d_val) : "r"(d_ptr) : "memory");

        // 요청 처리
        for (int i = 0; i < msg_hdr->n_reqs; ++i) {
            struct RequestHeader *req = message_header_get_request_ptr(msg_hdr, i);
            switch (req->type) {
                case REQUEST_TYPE_OP_COMPUTE:
                    req->state = execute_op_simple((struct OpComputeRequest *)req->data);
                    break;
                case REQUEST_TYPE_RPCMEM_MAP:
                    mmap_manager_put_map(map_req->fds[j]);
                    break;
            }
        }

        // release 시맨틱으로 완료 플래그 설정
        asm volatile("memd_rl(%0):at = %1" ::"r"(d_ptr), "r"(d_val) : "memory");
    }
}
```

---

## 4. 공유 라이브러리 로딩 메커니즘

### 4.1 ggml-hexagon: 이중 로딩 구조

분석 위치: [htp-drv.cpp](../../ggml/src/ggml-hexagon/htp-drv.cpp), [CMakeLists.txt](../../ggml/src/ggml-hexagon/CMakeLists.txt)

**Host측 (CPU)**:
```
htpdrv_init()
  → dlopen("libcdsprpc.dll" 또는 "libcdsprpc.so")
  → dlsym으로 20+ 심볼 바인딩 (rpcmem_*, dspqueue_*, remote_*, fastrpc_*)
```

**DSP측 (Hexagon)**:
```
htp_iface_open("file:///libggml-htp-v73.so?htp_iface_skel_handle_invoke")
  → FastRPC 프레임워크가 DSP에서 libggml-htp-v73.so 로딩
  → htp_iface_skel_handle_invoke가 IDL dispatch 핸들러
```

DSP skel은 **아키텍처별**로 빌드됩니다 (v68, v69, v73, v75, v79, v81):

```cmake
# 위치: ggml/src/ggml-hexagon/CMakeLists.txt:74-80
build_htp_skel(v68)
build_htp_skel(v69)
build_htp_skel(v73)
build_htp_skel(v75)
build_htp_skel(v79)
build_htp_skel(v81)
```

### 4.2 htp-ops-lib: Stub/Skel 분리

분석 위치: [htp-ops-lib/CMakeLists.txt](../../../htp-ops-lib/CMakeLists.txt)

```cmake
# HLOS (CPU): libhtp_ops.so (stub)
add_library(htp_ops SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_stub.c    # QAIC 자동 생성
    ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
    src/host/op_export.c
    src/host/session.c)

# DSP (Hexagon): libhtp_ops_skel.so
add_library(htp_ops_skel SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_skel.c    # QAIC 자동 생성
    src/dsp/commu.c
    src/dsp/hmx_mgr.c
    src/dsp/mmap_mgr.cc
    src/dsp/op_executor.cc
    src/dsp/ops/flash_attn.c
    src/dsp/ops/mat_mul.c
    src/dsp/ops/rms_norm.c
    ...)
```

빌드 산출물:
- `libhtp_ops.so` (AArch64, CPU에서 실행) — FastRPC stub
- `libhtp_ops_skel.so` (Q6DSP, Hexagon에서 실행) — FastRPC skeleton

---

## 5. RPC / 통신 구조 비교

### 5.1 ggml-hexagon: dspqueue 비동기 메시지 큐

```
CPU                                    DSP
┌──────────────┐                      ┌──────────────┐
│ dspqueue_write│──── 요청 큐 ───────▶│dspqueue_read  │
│ (htp_general_│   (128KB 공유메모리) │_noblock       │
│  req + bufs) │                      │               │
│              │                      │ htp_packet_   │
│ dspqueue_read│◀─── 응답 큐 ────────│ callback      │
│ (htp_general_│   (64KB 공유메모리)  │ dspqueue_write│
│  rsp)        │                      │               │
└──────────────┘                      └──────────────┘
```

**장점**: 
- 비동기 파이프라인: 여러 Op을 연속 enqueue하고 마지막에 flush
- 제로-카피: `dspqueue_buffer`에 FD/offset으로 ion 버퍼 참조
- 콜백 기반 처리: DSP가 유휴 상태일 때만 작업 수행

### 5.2 htp-ops-lib: 이중 통신 모드

#### Mode A: 직접 FastRPC 호출

```
CPU                                    DSP
┌──────────────┐                      ┌──────────────┐
│htp_ops_rms_  │── FastRPC invoke ──▶│htp_ops_rms_  │
│norm_f32()    │   (동기 호출)        │norm_f32()    │
│              │◀─ 리턴 ─────────────│              │
└──────────────┘                      └──────────────┘
```

#### Mode B: 공유 메모리 채널 (저지연)

```
CPU                                    DSP
┌──────────────┐                      ┌──────────────┐
│MessageHeader │ ── 공유 메모리 ──── │msg_receiver_ │
│.state.v[0]=1 │   (rpcmem 버퍼)     │loop()        │
│              │                      │ 폴링 + 처리  │
│while(v[1]!=1)│◀─ v[1]=1 ──────────│.state.v[1]=1 │
│              │   (release 시맨틱)   │              │
└──────────────┘                      └──────────────┘
```

---

## 6. Op 디스패치 방식 비교

### 6.1 ggml-hexagon Op 목록 (htp_op enum)

분석 위치: [htp/htp-msg.h](../../ggml/src/ggml-hexagon/htp/htp-msg.h#L52)

```c
enum htp_op {
    HTP_OP_MUL = 0, HTP_OP_ADD, HTP_OP_SUB, HTP_OP_DIV,
    HTP_OP_MUL_MAT, HTP_OP_MUL_MAT_ID,
    HTP_OP_RMS_NORM, HTP_OP_UNARY_SILU, HTP_OP_UNARY_GELU,
    HTP_OP_GLU_SWIGLU, HTP_OP_GLU_SWIGLU_OAI, HTP_OP_GLU_GEGLU,
    HTP_OP_SOFTMAX, HTP_OP_ADD_ID, HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT, HTP_OP_SET_ROWS, HTP_OP_GET_ROWS,
    HTP_OP_SCALE, HTP_OP_CPY, HTP_OP_ARGSORT,
    HTP_OP_SQR, HTP_OP_SQRT, HTP_OP_SUM_ROWS, HTP_OP_SSM_CONV,
};
```

### 6.2 htp-ops-lib Op 목록 (HtpOpsIndex)

분석 위치: [htp-ops-lib/include/op_reg.h](../../../htp-ops-lib/include/op_reg.h)

```c
enum HtpOpsIndex {
    HTP_OPS_RMS_NORM_F32,
    HTP_OPS_MAT_MUL_PERMUTED_W16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
    HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
    HTP_OPS_COUNT,
};
```

### 6.3 Op 기능 매핑

| ggml-hexagon Op | htp-ops-lib Op | 비고 |
|-----------------|----------------|------|
| `HTP_OP_RMS_NORM` | `HTP_OPS_RMS_NORM_F32` | 동일 기능, 다른 디스패치 |
| `HTP_OP_MUL_MAT` (Q4_0) | `HTP_OPS_MAT_MUL_PERMUTED_W4D16A32` | htp-ops-lib는 HMX 기반, Crouton 레이아웃 |
| `HTP_OP_MUL_MAT` (Q8_0) | `HTP_OPS_MAT_MUL_PERMUTED_W8D16A32` | 동일 |
| `HTP_OP_MUL_MAT` (F16) | `HTP_OPS_MAT_MUL_PERMUTED_W16A32` | htp-ops-lib는 FP16 HMX 사용 |
| `HTP_OP_FLASH_ATTN_EXT` | `HTP_OPS_FLASH_ATTN_QO_F32_KV_F16` | htp-ops-lib는 FP16 HMX FlashAttention |
| `HTP_OP_SOFTMAX`, `HTP_OP_ROPE`, `HTP_OP_ADD` 등 | ❌ 없음 | htp-ops-lib에는 해당 Op 없음 |
| ❌ 없음 | `HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL` | IQ4_NL 양자화 지원 (htp-ops-lib만) |

---

## 7. 공유 데이터 구조

### 7.1 htp-ops-lib의 호스트↔DSP 공유 구조체

분석 위치: [htp-ops-lib/include/message.h](../../../htp-ops-lib/include/message.h), [htp-ops-lib/include/op_reg.h](../../../htp-ops-lib/include/op_reg.h)

#### `message.h` - 공유 메모리 채널 프로토콜

```c
struct MessageState {
    union { volatile uint8_t v[8]; volatile uint64_t d; };
};  // v[0]: 호스트→DSP 트리거, v[1]: DSP→호스트 완료 플래그

struct MessageHeader {
    struct MessageState state;
    uint32_t checksum;
    int32_t  n_reqs;
    int32_t  req_offsets[0];    // 가변 길이 배열
};

struct RequestHeader {
    int32_t state;              // 요청 결과 상태 (-1: 에러, 0: 성공)
    int32_t type;               // REQUEST_TYPE_OP_COMPUTE, REQUEST_TYPE_RPCMEM_MAP
    uint8_t data[0];
};

enum RequestType {
    REQUEST_TYPE_NO_OP = 0,
    REQUEST_TYPE_RPCMEM_MAP,    // rpcmem FD 매핑 요청
    REQUEST_TYPE_OP_COMPUTE,    // Op 연산 요청
};

struct OpComputeRequest {
    uint32_t op;                // HtpOpsIndex 값
    uint8_t  payload[0];        // Op별 파라미터
};
```

#### `op_reg.h` - Op 파라미터 구조체

```c
struct RpcmemBufAddr {
    int32_t fd;                 // rpcmem FD
    int32_t offset;             // 버퍼 내 오프셋
};

struct RmsNormF32Params {
    struct RpcmemBufAddr dst, src;
    int32_t ne0, ne1;
};

struct MatMulParams {
    struct RpcmemBufAddr output, activation, weight;
    int32_t m, k, n;
};

struct FlashAttnParams {
    struct RpcmemBufAddr o, q, k, v, mask;
    int32_t qo_len, kv_len, n_heads, n_kv_heads, head_dim;
};
```

### 7.2 ggml-hexagon의 Host↔DSP 메시지 구조

분석 위치: [htp/htp-msg.h](../../ggml/src/ggml-hexagon/htp/htp-msg.h)

```c
struct htp_tensor {
    uint32_t data;              // 버퍼 오프셋/포인터
    uint32_t type;              // ggml_type 매핑
    uint32_t ne[4], nb[4];      // 차원 및 스트라이드
};

struct htp_general_req {
    uint32_t op;                // htp_op enum
    int32_t  op_params[16];     // Op별 파라미터 (RoPE freq 등)
    uint32_t flags;             // SKIP_QUANTIZE, EARLY_WAKEUP 등
    struct htp_tensor src0, src1, src2, src3, src4, dst;
};

struct htp_general_rsp {
    uint32_t op, status;
    uint32_t prof_usecs, prof_cycles, prof_pkts;
};
```

**차이점**: `ggml-hexagon`은 **범용 텐서 디스크립터** (`htp_tensor`)를 사용하여 임의의 Op을 기술할 수 있지만, `htp-ops-lib`는 **Op별 전용 파라미터** 구조체를 사용합니다.

---

## 8. 통합 가능성 및 차이점 요약

### 8.1 현재 상태

- **`ggml-hexagon` 백엔드는 `htp-ops-lib`를 참조하지 않습니다.** 두 시스템은 완전히 독립적입니다.
- 두 시스템은 동일한 하드웨어(Hexagon cDSP)를 타겟으로 하지만, **통신 방식, Op 등록 방식, 가중치 레이아웃이 모두 다릅니다.**

### 8.2 주요 차이점 정리

| 측면 | ggml-hexagon | htp-ops-lib |
|------|-------------|-------------|
| **통신** | dspqueue (비동기 큐, 콜백) | FastRPC (동기) + 공유 메모리 폴링 |
| **Op 등록** | 정적 switch-case | IDL 기반 개별 RPC 함수 |
| **가중치 형식** | q4x4x2, q8x4x2, mxfp4x4x2 | Crouton (FP16 HMX 레이아웃) |
| **HMX 사용** | 제한적 (v73+ 조건부) | 핵심 의존 (FP16 HMX 필수) |
| **DSP skel** | `libggml-htp-v{arch}.so` (다중 아키텍처) | `libhtp_ops_skel.so` (단일) |
| **Host stub** | `htp_iface_stub.c` (세션 관리만) | `htp_ops_stub.c` (세션 + Op) |
| **Op 범위** | 25+ 연산자 (완전한 LLM 추론) | 6개 (matmul, rms_norm, flash_attn) |
| **VTCM 관리** | HAP_compute_res 기반, 동적 해제 | vtcm_mgr.cc (독자 관리) |
| **워커 스레드** | worker-pool.c (dspqueue 콜백 내) | worker_pool.c (독립 초기화) |

### 8.3 통합 전략 옵션

#### 옵션 A: htp-ops-lib의 HMX 커널을 ggml-hexagon에 이식

- `htp-ops-lib`의 **`mat_mul.c`**, **`flash_attn.c`** 의 HMX 커널 코드를 `ggml-hexagon/htp/` 의 기존 Op 구현에 병합
- Crouton 레이아웃 ↔ q4x4x2 레이아웃 변환 레이어 추가
- dspqueue 기반 통신은 유지 (성능상 이점)

#### 옵션 B: htp-ops-lib를 별도 공유 라이브러리로 동적 로딩

- `ggml-hexagon`에서 `libhtp_ops.so`를 `dlopen`으로 로딩
- 특정 Op(특히 HMX 기반 matmul/flash_attn)만 htp-ops-lib에 위임
- 기존 dspqueue 파이프라인과 병행 운영

#### 옵션 C: 공유 메모리 채널 방식 도입

- `htp-ops-lib`의 `message.h`/`commu.c` 기반 **저지연 폴링 채널**을 ggml-hexagon에 추가
- dspqueue 오버헤드 제거 (특히 소규모 Op에 유효)
- 대신 CPU 폴링에 의한 전력 소모 증가 가능

### 8.4 공유 인프라 (이미 호환 가능)

다음 인프라는 두 시스템 모두 동일한 Hexagon SDK API를 사용하므로 호환 가능합니다:

- `rpcmem_alloc` / `rpcmem_to_fd` / `rpcmem_free`
- `fastrpc_mmap` / `fastrpc_munmap`
- `remote_handle64_open` / `remote_handle64_control` / `remote_session_control`
- VTCM 할당: `HAP_compute_res_*`
- HVX 워커 스레드 풀: `qurt_thread_*`
- DMA: `dma_queue_create`

---

## 부록: 소스 트리 내 "htp_ops" / "hexagon" / "fastrpc" / "rpcmem" 참조 위치

### ggml/ 내 참조

| 키워드 | 파일 | 기능 |
|--------|------|------|
| `rpcmem` | `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | 버퍼 할당/해제/FD 변환 |
| `rpcmem` | `ggml/src/ggml-hexagon/htp-drv.cpp` | 함수 포인터 로딩 |
| `rpcmem` | `ggml/src/ggml-hexagon/htp-drv.h` | 인클루드 선언 |
| `fastrpc` | `ggml/src/ggml-hexagon/htp-drv.cpp` | mmap/munmap 함수 로딩 |
| `fastrpc` | `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | 버퍼 매핑 |
| `dspqueue` | `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | Op 전송/수신 |
| `dspqueue` | `ggml/src/ggml-hexagon/htp/main.c` | DSP 콜백 처리 |
| `hexagon` | `ggml/include/ggml-hexagon.h` | 퍼블릭 API |
| `htp_ops` | `ggml/src/ggml-hexagon/htp/htp-ops.h` | DSP Op 인터페이스 (**htp-ops-lib와 무관**) |

### htp-ops-lib/ 내 참조

| 키워드 | 파일 | 기능 |
|--------|------|------|
| `htp_ops` | `htp-ops-lib/include/htp_ops.idl` | FastRPC IDL 정의 |
| `htp_ops` | `htp-ops-lib/src/host/op_export.c` | RPC 래퍼 |
| `htp_ops` | `htp-ops-lib/src/host/session.c` | 세션 열기/닫기 |
| `rpcmem` | `htp-ops-lib/src/host/test.c` | 테스트 버퍼 할당 |
| `fastrpc` | `htp-ops-lib/src/host/test.c` | 버퍼 매핑 |
| `fastrpc` | `htp-ops-lib/src/host/session.c` | QoS 설정 |

> **참고**: `ggml/src/ggml-hexagon/htp/htp-ops.h`의 `htp_ops_context`는 `htp-ops-lib`의 `HtpOpsIndex`와는 **완전히 별개의 코드**입니다. 이름이 유사하지만 구조와 목적이 다릅니다.
