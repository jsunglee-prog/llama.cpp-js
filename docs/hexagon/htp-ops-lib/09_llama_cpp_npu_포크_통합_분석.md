# 09. llama.cpp-npu 포크의 htp-ops-lib 통합 방식 분석

> 분석 대상: `C:\Users\pando\source\llama.cpp-npu\ggml\src\ggml-htp\` (12개 파일)

## 1. 핵심 결론: .so를 직접 포함하지 않는다

**llama.cpp-npu 포크는 `libhtp_ops.so`를 빌드 시 링크하지 않는다.** 대신:

1. **런타임에 `dlopen`으로 동적 로드**한다
2. **프로토콜 헤더 2개**(`message.h`, `op_reg.h`)를 htp-ops-lib에서 **복사**하여 소스에 포함한다
3. **FastRPC 함수들**도 SDK 헤더 없이 **직접 선언 + dlopen으로 로드**한다

## 2. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│  llama.cpp-npu (ARM CPU)                                        │
│                                                                 │
│  ggml-htp.cc          htp-ops.cc          dsprpc_interface.cc   │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │ggml 백엔드│───▶│OP 디스패치    │    │ FastRPC 래퍼       │     │
│  │등록/초기화│    │파라미터 직렬화│    │                    │     │
│  └──────────┘    │공유메모리 폴링│    │ dlopen             │     │
│       │          └───────┬──────┘    │ "libcdsprpc.so"    │     │
│       │                  │           └────────┬───────────┘     │
│       │   dlopen         │                    │                 │
│       │   "libhtp_ops.so"│          rpcmem_alloc/free/to_fd    │
│       ▼                  │          fastrpc_mmap/munmap         │
│  ┌──────────┐            │                    │                 │
│  │session.c │────────────┘                    │                 │
│  │함수 호출 │     shared memory (4KB)         │                 │
│  └──────────┘         │                       │                 │
│                       │  ┌────────────────────┘                 │
│                       │  │                                      │
├───────────────────────┼──┼──────────────────────────────────────┤
│  FastRPC boundary     │  │                                      │
├───────────────────────┼──┼──────────────────────────────────────┤
│  Hexagon DSP (CDSP)   │  │                                      │
│                       ▼  ▼                                      │
│  ┌──────────────────────────┐                                   │
│  │ libhtp_ops_skel.so       │                                   │
│  │ • commu.c (폴링 루프)    │                                   │
│  │ • mat_mul.c (HMX 커널)   │                                   │
│  │ • flash_attn.c           │                                   │
│  │ • rms_norm.c             │                                   │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 동적 라이브러리 로딩 구조

### 3.1 libcdsprpc.so 로딩 — FastRPC 런타임

**파일**: `dsprpc_interface.cc`

```cpp
// Qualcomm FastRPC 라이브러리를 런타임에 dlopen
void * load_lib() {
    auto * lib = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        GGML_ABORT("unable to load libcdsprpc.so");
    }
    return lib;
}
```

SDK 헤더를 포함하지 않고 `dsprpc_interface.h`에서 **모든 함수 프로토타입을 직접 선언**:

```c
// SDK 없이 직접 선언한 FastRPC 타입과 함수
enum fastrpc_map_flags { ... };
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_FLAG_UNCACHED  1
#define CDSP_DOMAIN_ID 3

void rpcmem_init(void);
void * rpcmem_alloc(int heap_id, uint32_t flags, int size);
int rpcmem_to_fd(void * p);
int fastrpc_mmap(int domain, int fd, void * addr, int offset, 
                 size_t length, enum fastrpc_map_flags flags);
```

**싱글톤 패턴**으로 함수 포인터를 한 번만 로드하고 캐싱:

```cpp
struct dsprpc_interface {
    rpcmem_alloc_t * rpcmem_alloc_fn = 
        reinterpret_cast<rpcmem_alloc_t *>(load_fn("rpcmem_alloc"));
    // ... 7개 함수 모두 동일 패턴
    
    static dsprpc_interface * instance() {
        static dsprpc_interface * _instance = new dsprpc_interface;
        return _instance;
    }
};
```

### 3.2 libhtp_ops.so 로딩 — DSP 세션 관리

**파일**: `ggml-htp-impl.h`, `ggml-htp.cc`

```cpp
// 경로 상수 정의
static const char * HTP_OPS_DL_PATH = "libhtp_ops.so";

// 초기화 시 dlopen
ggml_backend_htp_context::ggml_backend_htp_context() 
    : mapper(3 * 1024UL * 1024 * 1024, true) 
{
    rpcmem_init();
    
    ops_dl_handle = dlopen(HTP_OPS_DL_PATH, RTLD_LAZY | RTLD_LOCAL);
    if (ops_dl_handle != nullptr) {
        // dlsym으로 3개 함수만 가져옴
        auto open_session = dlsym(ops_dl_handle, "open_dsp_session");
        auto init_htp_ops = dlsym(ops_dl_handle, "init_htp_backend");
        
        open_session(CDSP_DOMAIN_ID, 1);  // DSP 세션 열기
        init_htp_ops();                    // HMX/VTCM 초기화
        init_message_channel();            // 공유 메모리 채널 설정
    }
}
```

**즉, `libhtp_ops.so`에서 사용하는 함수는 단 4개뿐:**

| 함수 | 출처 (htp-ops-lib) | 용도 |
|------|---------------------|------|
| `open_dsp_session()` | `src/host/session.c` | DSP 세션 열기 |
| `init_htp_backend()` | `src/host/session.c` | HMX/VTCM 초기화 |
| `create_htp_message_channel()` | `src/host/session.c` | 공유 메모리 채널 생성 |
| `close_dsp_session()` | `src/host/session.c` | DSP 세션 종료 |

> **연산 커널 함수(`htp_ops_rpc_mat_mul_*` 등)는 사용하지 않는다!**
> 대신 공유 메모리 폴링 방식(`message.h` 프로토콜)으로만 연산을 디스패치한다.

## 4. 프로토콜 헤더 복사

### 4.1 동일한 헤더 2개

포크의 `ggml-htp/` 디렉토리에 있는 다음 파일들은 `htp-ops-lib/include/`의 것과 **바이트 단위로 동일**:

| 포크 파일 | 원본 파일 |
|-----------|-----------|
| `ggml-htp/message.h` | `htp-ops-lib/include/message.h` |
| `ggml-htp/op_reg.h` | `htp-ops-lib/include/op_reg.h` |

이 2개 헤더가 CPU↔DSP 간 공유 메모리 프로토콜의 **계약(contract)**:

```
message.h  →  MessageHeader, RequestHeader, OpComputeRequest 등 통신 구조체
op_reg.h   →  HtpOpsIndex 열거형, MatMulParams, FlashAttnParams 등 파라미터 구조체
```

### 4.2 왜 복사하는가?

- htp-ops-lib는 별도 프로젝트이므로 CMake `target_link_libraries`로 직접 연결 불가
- SDK 의존성 없이 빌드하기 위해 필요한 최소 인터페이스만 복사
- ABI 호환성만 맞으면 되므로 바이너리 수준에서 동작

## 5. 연산 디스패치: 공유 메모리 폴링 방식

### 5.1 htp_ops_compute_op() 흐름

**파일**: `htp-ops.cc`

RPC 호출 대신 **공유 메모리 메시지 기반 디스패치**를 사용한다:

```
1. 파라미터 직렬화
   ├─ ggml_tensor에서 fd/offset 추출
   ├─ MatMulParams/FlashAttnParams 구조체에 기록
   └─ param_buf[4096]에 복사

2. 메시지 작성
   ├─ MessageHeader에 요청 수, 오프셋 기록
   ├─ RequestHeader + OpComputeRequest + payload 직렬화
   └─ 체크섬 계산

3. 요청 발행
   └─ atomic store로 state.v[0] = 1 (memory_order_release)

4. 응답 대기
   └─ while(state.v[1] == 0) usleep(1); (폴링)

5. 결과 확인
   └─ RequestHeader.state 반환
```

### 5.2 RPC vs 폴링 모드 선택

코드에 두 가지 경로가 모두 구현되어 있다:

```cpp
constexpr bool prefer_rpc = false;  // 하드코딩: 폴링 모드 사용

if (prefer_rpc) {
    // 기존 FastRPC 호출 (느림)
    auto op_fn = dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_permuted_w16a32");
    return op_fn(output_fd, ..., m, k, n);
}

// 폴링 모드 (빠름) — 실제 사용되는 경로
op_index = HTP_OPS_MAT_MUL_PERMUTED_W16A32;
// → 공유 메모리에 기록 후 폴링
```

**폴링 모드를 기본으로 사용하는 이유:**
- FastRPC 호출의 커널 왕복 오버헤드(~10-100μs) 제거
- 공유 메모리 폴링은 수 μs 수준의 지연시간

## 6. ggml 백엔드 등록 구조

### 6.1 백엔드 계층 구조

**파일**: `ggml-htp.cc`

```
ggml_backend_reg (MyHTP)
  └── ggml_backend_device (HTP, ACCEL 타입)
        └── ggml_backend (MyHTP)
              ├── buffer_type: RPCMEM (rpcmem_alloc 기반)
              └── graph_compute: CPU+HTP 하이브리드
```

### 6.2 하이브리드 실행 모델

```cpp
static enum ggml_status ggml_backend_htp_graph_compute(
    ggml_backend_t backend, struct ggml_cgraph * cgraph) 
{
    // CPU 실행 계획 생성
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, ctx->n_threads, ctx->threadpool);
    
    // 하이브리드 실행: CPU op + HTP op 혼합 실행
    return ggml_graph_compute_htp_hybrid(cgraph, &cplan);
}
```

`ggml_graph_compute_htp_hybrid()`는 [htp-cpu-impl.c](htp-cpu-impl.c) (13,139줄)에 구현되어 있으며,
이는 **ggml-cpu의 `ggml_compute_forward()`를 수정한 버전**이다:

- 각 노드에 대해 `htp_ops_support_op()`으로 HTP 지원 여부 확인
- 지원되면 `htp_ops_compute_op()`으로 DSP에 디스패치
- 미지원이면 CPU 구현으로 폴백

### 6.3 RPCMEM 버퍼 타입

모든 텐서 데이터를 `rpcmem_alloc()`으로 할당:

```cpp
static ggml_backend_buffer_t ggml_backend_htp_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) 
{
    // Qualcomm 공유 메모리 할당 (CPU ↔ DSP 간 zero-copy)
    void * data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
    return ggml_backend_buffer_init(buft, ggml_backend_htp_buffer_i, data, size);
}
```

### 6.4 RpcMemMapper — LRU 매핑 관리

**파일**: `rpcmem_mapper.cc`, `rpcmem_mapper.h`

DSP에서 직접 접근하려면 `fastrpc_mmap`으로 매핑해야 하지만, 동시 매핑 한도가 있다.
이를 LRU(Least Recently Used) 캐시로 관리:

```cpp
RpcMemMapper mapper(3 * 1024UL * 1024 * 1024, true);  // 3GB 매핑 한도

// validate() 호출 시:
// 1. 이미 매핑된 버퍼 → LRU 맨 앞으로 이동
// 2. 새 버퍼 → 공간 부족 시 LRU 끝의 매핑 해제 후 새로 매핑
// 3. defer_unmap=true → unmap을 다음 compute 시 DSP에 알림 후 실행
```

## 7. 지원 연산 및 조건

**파일**: `htp-ops.cc` → `htp_ops_support_op()`

| 연산 | weight 타입 | activation 타입 | 출력 타입 | 추가 조건 |
|------|-------------|-----------------|-----------|-----------|
| MUL_MAT | F16 | F32 | F32 | K%32==0, N%32==0 |
| MUL_MAT | Q4_0 | F32 | F32 | K%32==0, N%32==0 |
| MUL_MAT | Q8_0 | F32 | F32 | K%32==0, N%32==0 |
| MUL_MAT | IQ4_NL | F32 | F32 | K%32==0, N%32==0 |
| FLASH_ATTN_EXT | Q=F32, KV=F16 | mask=F16 | F32 | max_bias==0, logit_softcap==0 |
| RMS_NORM | F32 | F32 | F32 | **비활성화** (`return false`) |

> RMS_NORM은 코드에 있지만 `return false`로 하드코딩되어 비활성화 상태

## 8. 파일별 역할 요약

| 파일 | 줄 수 | 역할 |
|------|-------|------|
| `ggml-htp.cc` | 459 | ggml 백엔드 등록, RPCMEM 버퍼 타입, 하이브리드 그래프 실행 |
| `ggml-htp-impl.h` | 43 | 싱글톤 컨텍스트 구조체, `HTP_OPS_DL_PATH` 정의 |
| `htp-ops.cc` | 382 | OP 지원 판단, 파라미터 직렬화, 공유 메모리 폴링 디스패치 |
| `htp-ops.h` | 18 | `htp_ops_support_op()`, `htp_ops_compute_op()` 인터페이스 |
| `htp-cpu-impl.c` | 13,139 | ggml-cpu 수정본: CPU/HTP 하이브리드 forward pass |
| `dsprpc_interface.cc` | 88 | `libcdsprpc.so` dlopen + 래퍼 함수 |
| `dsprpc_interface.h` | 100 | FastRPC/rpcmem 타입 및 함수 선언 (SDK 불필요) |
| `rpcmem_mapper.cc` | 208 | LRU 기반 fastrpc_mmap 매핑 관리 |
| `rpcmem_mapper.h` | 53 | RpcMemMapper 클래스 선언 |
| `message.h` | 63 | 공유 메모리 통신 프로토콜 (htp-ops-lib와 동일) |
| `op_reg.h` | 54 | 연산 인덱스 + 파라미터 구조체 (htp-ops-lib와 동일) |
| `CMakeLists.txt` | 22 | 빌드 설정, libhtp_ops.so 미링크 |

## 9. 통합 패턴 정리

```
┌─────────────────────────────────────────────────┐
│          빌드 시 (Compile-time)                  │
│                                                  │
│  포함 방식:                                      │
│  ✗ libhtp_ops.so 정적/동적 링크 → 안 한다        │
│  ✗ htp-ops-lib CMake 연결 → 안 한다              │
│  ✓ message.h / op_reg.h 헤더 복사 → 한다         │
│  ✓ FastRPC 타입 직접 선언 → 한다 (SDK 불필요)    │
│                                                  │
├─────────────────────────────────────────────────┤
│          실행 시 (Runtime)                        │
│                                                  │
│  1) dlopen("libcdsprpc.so")                      │
│     └─ rpcmem_*, fastrpc_* 함수 로드             │
│                                                  │
│  2) dlopen("libhtp_ops.so")                      │
│     └─ open_dsp_session() 호출                   │
│     └─ init_htp_backend() 호출                   │
│     └─ create_htp_message_channel() 호출         │
│                                                  │
│  3) 연산 실행:                                    │
│     └─ 공유 메모리에 파라미터 기록               │
│     └─ atomic flag 설정                          │
│     └─ DSP 폴링 응답 대기                        │
│                                                  │
│  4) 종료:                                        │
│     └─ close_dsp_session() 호출                  │
│     └─ dlclose(libhtp_ops.so)                    │
└─────────────────────────────────────────────────┘
```

## 10. 배포 시 필요한 파일

Android 디바이스에서 실행하려면 다음 파일이 필요:

```
/vendor/lib64/libcdsprpc.so         ← 시스템 제공 (Qualcomm BSP)
/data/local/tmp/
  ├── llama-cli                     ← llama.cpp-npu 빌드 바이너리
  ├── libhtp_ops.so                 ← htp-ops-lib stub (ARM용)
  └── libhtp_ops_skel.so            ← htp-ops-lib skel (Hexagon DSP용)
      (또는 /vendor/lib/rfsa/adsp/)
```

`libhtp_ops.so`와 `libhtp_ops_skel.so`는 **별도로 빌드**하여 디바이스에 배치해야 한다.
포크의 빌드 시스템은 이 라이브러리를 생성하지 않으며, htp-ops-lib의 빌드 시스템에서 만들어야 한다.

## 11. 기존 ggml-hexagon과의 차이점 요약

| 항목 | ggml-hexagon (기존) | ggml-htp (포크) |
|------|---------------------|-----------------|
| DSP 통신 | dspqueue (비동기) | 공유 메모리 폴링 |
| DSP 라이브러리 | libggml-htp-v{arch}.so | libhtp_ops_skel.so |
| SDK 의존성 | HTP SDK 헤더 필요 | **SDK 불필요** (dlopen+직접 선언) |
| HMX 사용 | ✗ (HVX만) | ✓ (HMX FP16) |
| 지원 연산 수 | 25+ | 3 (MUL_MAT, FLASH_ATTN, RMS_NORM¹) |
| 텐서 메모리 | ION/dma-buf | rpcmem (UNCACHED) |
| 연산 경로 | 모든 op을 DSP에서 실행 | CPU+HTP 하이브리드 |
| 빌드 방식 | CMake native | CMake + 별도 skel 빌드 |

¹ RMS_NORM은 코드 존재하나 현재 비활성화
