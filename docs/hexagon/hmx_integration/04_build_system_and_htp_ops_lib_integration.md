# 빌드 시스템 및 htp-ops-lib 통합 분석

> 분석 파일:  
> - [ggml/src/ggml-hexagon/CMakeLists.txt](../../ggml/src/ggml-hexagon/CMakeLists.txt)  
> - [ggml/src/ggml-hexagon/htp/CMakeLists.txt](../../ggml/src/ggml-hexagon/htp/CMakeLists.txt)  
> - [htp-ops-lib/](../../htp-ops-lib/) 디렉토리  

---

## 1. Host 측 빌드 구조

**파일**: [ggml/src/ggml-hexagon/CMakeLists.txt](../../ggml/src/ggml-hexagon/CMakeLists.txt)

### Host 라이브러리 (`ggml-hexagon`):
```cmake
ggml_add_backend_library(ggml-hexagon
    ggml-hexagon.cpp
    htp-drv.cpp
    htp-drv.h
    libdl.h
    ../../include/ggml-hexagon.h)

target_link_libraries(ggml-hexagon PRIVATE htp_iface)
```

### IDL 스텁:
```cmake
add_library(htp_iface OBJECT
    ${CMAKE_CURRENT_BINARY_DIR}/htp_iface_stub.c)

build_idl(htp/htp_iface.idl htp_iface)
```

- `build_idl()`: Hexagon SDK의 CMake 함수로 `.idl` → `*_stub.c` (Host) + `*_skel.c` (DSP) 생성
- 생성된 stub 코드가 Host에서 FastRPC 호출을 담당

---

## 2. DSP 측 빌드 구조 (skel)

**파일**: [ggml/src/ggml-hexagon/htp/CMakeLists.txt](../../ggml/src/ggml-hexagon/htp/CMakeLists.txt)

### skel 라이브러리 (`ggml-htp-{version}`):
```cmake
add_library(${HTP_LIB} SHARED
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
)
```

### 크로스 컴파일:
- `ExternalProject_Add`를 통해 Hexagon 툴체인으로 크로스 컴파일
- 툴체인 파일: [htp/cmake-toolchain.cmake](../../ggml/src/ggml-hexagon/htp/cmake-toolchain.cmake)
- 빌드 대상 아키텍처: **v68, v69, v73, v75, v79, v81**

### 빌드 아키텍처별 skel 생성:
```cmake
build_htp_skel(v68)
build_htp_skel(v69)
build_htp_skel(v73)
build_htp_skel(v75)
build_htp_skel(v79)
build_htp_skel(v81)
```

각각 `libggml-htp-v{XX}.so` 파일로 생성됨.

### 컴파일 정의:
```cmake
target_compile_definitions(${HTP_LIB} PRIVATE
    NDEBUG=1                                         # 또는 HTP_DEBUG=1
    FP32_QUANTIZE_GROUP_SIZE=${GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE})
```

---

## 3. htp-ops-lib 구조 분석

### 디렉토리 구조:
```
htp-ops-lib/
├── include/
│   ├── dsp/                      ← DSP 측 헤더
│   │   ├── ops.h                 ← Op 커널 함수 선언
│   │   ├── hmx_mgr.h             ← HMX 매니저
│   │   ├── hmx_utils.h           ← HMX 유틸리티
│   │   ├── hvx_convert.h         ← HVX 변환 함수
│   │   ├── vtcm_mgr.h            ← VTCM 매니저
│   │   ├── dma_utils.h           ← DMA 유틸리티
│   │   ├── worker_pool.h         ← 워커 풀
│   │   ├── quants.h              ← 양자화 타입
│   │   ├── mmap_mgr.h            ← 메모리 매핑
│   │   └── ...
│   ├── host/                     ← Host 측 헤더
│   │   ├── session.h
│   │   └── op_export.h
│   ├── message.h                 ← 메시지 프로토콜
│   ├── op_reg.h                  ← Op 등록 열거형/구조체
│   └── htp_ops.idl               ← 별도 IDL
├── src/
│   ├── dsp/
│   │   ├── ops/
│   │   │   ├── mat_mul.c         ← HMX matmul 구현
│   │   │   ├── flash_attn.c      ← HMX flash attention
│   │   │   ├── flash_attn_sp_hdim.c
│   │   │   ├── rms_norm.c        ← HVX rms_norm
│   │   │   ├── mm_benchmark.c    ← 벤치마크
│   │   │   └── precompute_table.c
│   │   ├── hmx_mgr.c
│   │   ├── vtcm_mgr.cc
│   │   ├── mmap_mgr.cc
│   │   ├── op_executor.cc
│   │   ├── worker_pool.c
│   │   ├── commu.c
│   │   └── power.c
│   └── host/
│       ├── session.c
│       ├── op_export.c
│       └── test.c
└── CMakeLists.txt
```

### htp-ops-lib의 Op 목록 ([include/op_reg.h](../../htp-ops-lib/include/op_reg.h)):

```c
enum HtpOpsIndex {
    HTP_OPS_RMS_NORM_F32,                    // RMS Norm (HVX)
    HTP_OPS_MAT_MUL_PERMUTED_W16A32,        // F16 weight × F32 activation (HMX)
    HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,      // Q4_0 dequant → HMX matmul
    HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,      // Q8_0 dequant → HMX matmul
    HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL, // IQ4_NL dequant → HMX matmul
    HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,       // Flash Attention (HMX)
    HTP_OPS_COUNT,
};
```

### htp-ops-lib의 커널 함수 ([include/dsp/ops.h](../../htp-ops-lib/include/dsp/ops.h)):

```c
// RMS Norm (HVX 기반)
int hvx_rms_norm_f32(float *dst, const float *src, int ne0, int ne1);

// HMX MatMul 변형들
int hmx_mat_mul_permuted_w16a32(float *dst, const float *activation,
                                 const __fp16 *permuted_weight, int m, int k, int n);
int hmx_mat_mul_permuted_qk_0_d16a32(float *dst, const float *activation,
                                      const uint8_t *permuted_weight,
                                      int m, int k, int n, enum ggml_type weight_type);

// Flash Attention
int simple_flash_attn(__fp16 *O, const __fp16 *Q, const __fp16 *K, const __fp16 *V,
                      const __fp16 *mask, int qo_len, int kv_len,
                      int n_heads, int n_kv_heads, int head_dim);
int naive_flash_attn(float *O, const float *Q, const __fp16 *K, const __fp16 *V,
                     const __fp16 *mask, int qo_len, int kv_len,
                     int n_heads, int n_kv_heads, int head_dim);
```

### htp-ops-lib의 메시지 프로토콜 ([include/message.h](../../htp-ops-lib/include/message.h)):

htp-ops-lib는 자체 메시지 프로토콜(`MessageHeader` + `RequestHeader`)을 사용함.
이는 ggml-hexagon의 `htp_general_req`/`htp_general_rsp`와 **완전히 다른 포맷**.

### htp-ops-lib의 파라미터 구조체 ([include/op_reg.h](../../htp-ops-lib/include/op_reg.h)):

```c
struct MatMulParams {
    struct RpcmemBufAddr output;     // (fd, offset) 쌍
    struct RpcmemBufAddr activation;
    struct RpcmemBufAddr weight;
    int32_t m, k, n;
};

struct FlashAttnParams {
    struct RpcmemBufAddr o, q, k, v, mask;
    int32_t qo_len, kv_len, n_heads, n_kv_heads, head_dim;
};
```

---

## 4. 통합 시 아키텍처 차이점 및 제약사항

### 4.1 메시지 프로토콜 차이

| 항목 | ggml-hexagon | htp-ops-lib |
|------|-------------|-------------|
| 통신 방식 | `dspqueue` (비동기 큐) | `MessageHeader` (공유 메모리 기반?) |
| 텐서 기술 | `htp_tensor` (ne[], nb[], type) | `RpcmemBufAddr` (fd, offset) |
| Op 파라미터 | `op_params[16]` 배열 | Op별 전용 구조체 |
| 캐시 관리 | `dspqueue_buffer.flags` | 별도 관리 |
| 응답 | `htp_general_rsp` (상태+프로파일) | `MessageState` |

### 4.2 VTCM/리소스 관리 차이

| 항목 | ggml-hexagon | htp-ops-lib |
|------|-------------|-------------|
| VTCM | `vtcm_acquire/release` (proc_req에서) | `vtcm_mgr.h` (별도 매니저) |
| HMX | `HAP_compute_res`로 함께 획득 | `hmx_mgr.h` (별도 매니저) |
| 워커 풀 | `worker-pool.c` | `worker_pool.h/c` (별도 구현) |

### 4.3 가중치 포맷 차이

| 항목 | ggml-hexagon | htp-ops-lib |
|------|-------------|-------------|
| Q4_0 | Q4x4x2 repack (HVX 최적화) | permuted weight (HMX 최적화) |
| F16 | 원본 유지 | permuted weight |
| 전처리 | Host에서 repack | 별도 `precompute_table.c` |

---

## 5. HMX Op 통합 전략

### 방안 A: htp-ops-lib 커널만 가져와서 ggml-hexagon 프레임워크에 통합

**장점**: 기존 메시지 프로토콜, 큐, 빌드 시스템 재사용  
**필요 작업**:

1. **htp-ops-lib의 커널 함수를 ggml-hexagon skel에 직접 포함**
   - `hmx_mat_mul_permuted_w16a32()`, `simple_flash_attn()` 등을 skel 빌드에 포함
   - 의존하는 `hmx_mgr.c`, `vtcm_mgr.cc`, `dma_utils.h` 등도 포함

2. **어댑터 Op 작성**
   - 기존 `proc_matmul_req()` 패턴을 따르는 `proc_hmx_matmul_req()` 작성
   - `htp_ops_context`에서 텐서 정보를 추출하여 htp-ops-lib 커널의 파라미터 형식으로 변환

3. **가중치 repack 전략 결정**
   - htp-ops-lib는 "permuted weight" 포맷을 사용하므로 별도 repack 필요
   - 기존 `repack_buffer_type`에 HMX용 repack 추가 가능

4. **빌드 시스템 수정**
   - `htp/CMakeLists.txt`에 htp-ops-lib 소스 파일 추가
   - include 경로 추가

### 방안 B: htp-ops-lib를 별도 공유 라이브러리로 빌드하고 skel에서 링크

**장점**: htp-ops-lib 코드 수정 최소화  
**단점**: 빌드 복잡도 증가, 두 개의 VTCM/HMX 매니저 충돌 가능

### **권장**: 방안 A (커널 함수 직접 통합)

---

## 6. 파일별 수정 위치 요약

### 새 Op 추가 체크리스트:

| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `htp/htp-msg.h` | `enum htp_op`에 새 HMX Op 코드 추가 |
| 2 | `htp/htp-ops.h` | `int op_hmx_matmul(struct htp_ops_context *);` 선언 추가 |
| 3 | `htp/hmx-matmul-ops.c` (새 파일) | HMX matmul Op 커널 구현 |
| 4 | `htp/main.c` | `htp_packet_callback()` switch에 case 추가 + `proc_hmx_matmul_req()` 작성 |
| 5 | `htp/CMakeLists.txt` | 새 소스 파일 + htp-ops-lib 소스/헤더 추가 |
| 6 | `ggml-hexagon.cpp` | `supports_op`에 분기 추가, `init_hmx_matmul_req()` 작성, `graph_compute`에 dispatch 추가 |
| 7 | `ggml-hexagon.cpp` | (옵션) HMX용 가중치 repack 함수 추가 |
| 8 | `CMakeLists.txt` (상위) | htp-ops-lib include path 추가 |

### 기존 Op에서 HMX 버전으로 전환할 경우:
- `ggml_hexagon_supported_mul_mat()`에서 HMX 능력 감지 후 HMX Op으로 라우팅
- 동일한 `GGML_OP_MUL_MAT`에 대해 HVX vs HMX 분기
- 또는 별도의 `HTP_OP_HMX_MUL_MAT` Op 코드 사용

