# HMX Op 추가를 위한 통합 가이드 및 제약사항

> 이 문서는 `htp-ops-lib`의 HMX 기반 연산을 `ggml-hexagon` 백엔드에 추가하기 위한 구체적인 가이드를 제공합니다.

---

## 1. End-to-End 데이터 흐름 (예: MUL_MAT)

```
[Host CPU]                                [DSP HTP]
                                          
1. supports_op() → true                   
2. graph_compute():                       
   ├─ init_binary_req():                  
   │   ├─ req.op = HTP_OP_MUL_MAT        
   │   ├─ htp_req_tensor_init(src0, src1) 
   │   └─ htp_req_buff_init(bufs[])       
   │                                       
   ├─ sess->enqueue():                    
   │   ├─ op_pending++                    
   │   └─ dspqueue_write(req, bufs)  ────→  htp_packet_callback():
   │                                        ├─ dspqueue_read_noblock()
   │                                        ├─ switch(req.op) → proc_matmul_req():
   │                                        │   ├─ octx 구성  
   │                                        │   ├─ bufs[N].ptr → octx.src*.data
   │                                        │   ├─ vtcm_acquire()
   │                                        │   ├─ op_matmul(&octx) ← HVX 커널
   │                                        │   │   ├─ DMA: DDR → VTCM
   │                                        │   │   ├─ HVX vec_dot 연산
   │                                        │   │   └─ DMA: VTCM → DDR
   │                                        │   ├─ vtcm_release()
   │                                        │   └─ send_htp_rsp() ────→
   └─ flush():                    ←────────  dspqueue_write(rsp)
       ├─ dspqueue_read(&rsp)               
       ├─ rsp.status 확인                   
       └─ op_pending--                      
```

---

## 2. 제약사항 정리

### 2.1 메시지 크기 제한
- `HTP_MAX_MESSAGE_SIZE = sizeof(htp_general_req)` — 약 340 bytes
- 텐서 최대 **5개 src + 1개 dst** (src0~src4, dst)
- 버퍼 최대 **8개** (`HTP_MAX_PACKET_BUFFERS = 8`)

### 2.2 데이터 타입 제한
`enum htp_data_type` ([htp-msg.h](../../ggml/src/ggml-hexagon/htp/htp-msg.h#L38-L48)):
- F32, F16, Q4_0, Q8_0, I32, I64, MXFP4 만 지원
- **htp-ops-lib와의 차이**: htp-ops-lib는 `ggml_type`을 직접 사용하여 IQ4_NL 등도 지원

### 2.3 VTCM 크기 제한
- 기본 8MB VTCM ([main.c L202](../../ggml/src/ggml-hexagon/htp/main.c#L202))
- matmul의 경우 `src0->ne[1] > 16 * 1024`이면 지원 안 함 (VTCM에 안 들어감)
- HMX Op의 tiling 전략이 VTCM 크기에 맞아야 함

### 2.4 Repack 제약
- 양자화 가중치는 반드시 `repack_buffer_type`으로 할당된 버퍼에 있어야 함
- HMX matmul은 "permuted weight" 포맷을 요구 → 별도 repack 전략 필요
- ggml-hexagon 기존 repack: Q4x4x2, Q8x4x2, MXFP4x4x2

### 2.5 HMX 아키텍처 제약
- HMX는 v73 이상에서만 사용 가능
- `htp_iface_open()`에서 HMX 전원은 켜지만 아키텍처 체크는 별도 필요
- `get_hex_arch_ver()`로 아키텍처 버전 확인 가능 ([htp-drv.h](../../ggml/src/ggml-hexagon/htp-drv.h#L99))

### 2.6 동시 리소스 사용
- VTCM + HMX가 `HAP_compute_res`로 함께 관리됨
- `vtcm_acquire()` 시 HMX도 함께 획득되므로 별도 HMX lock 불필요
- 다만 htp-ops-lib의 `hmx_mgr.c`가 별도 HMX 관리를 하므로 **충돌 방지 필요**

---

## 3. HMX MatMul 통합 예시 (의사코드)

### Step 1: `htp-msg.h` — Op 코드 추가

```c
enum htp_op {
    // ... 기존 Op들 ...
    HTP_OP_SSM_CONV,
    HTP_OP_HMX_MUL_MAT,         // 새로 추가
    HTP_OP_HMX_FLASH_ATTN,      // 새로 추가
    INVALID
};
```

### Step 2: `htp-ops.h` — 커널 선언 추가

```c
int op_hmx_matmul(struct htp_ops_context * octx);
int op_hmx_flash_attn(struct htp_ops_context * octx);
```

### Step 3: `hmx-matmul-ops.c` — 커널 구현

```c
#include "htp-ops.h"
#include "dsp/ops.h"  // htp-ops-lib의 커널 함수

int op_hmx_matmul(struct htp_ops_context * octx) {
    float * dst_data = (float *) octx->dst.data;
    float * act_data = (float *) octx->src1.data;   // activation
    void  * wgt_data = (void *)  octx->src0.data;   // weight (permuted)

    int m = octx->src1.ne[1];  // batch
    int k = octx->src0.ne[0];  // inner dim
    int n = octx->src0.ne[1];  // output dim

    // htp-ops-lib의 HMX matmul 호출
    return hmx_mat_mul_permuted_w16a32(dst_data, act_data,
                                        (__fp16 *) wgt_data, m, k, n);
}
```

### Step 4: `main.c` — 패킷 콜백에 case 추가

```c
case HTP_OP_HMX_MUL_MAT:
    if (n_bufs != 3) {
        FARF(ERROR, "Bad hmx-matmul-req buffer list");
        continue;
    }
    proc_matmul_req(ctx, &req, bufs, n_bufs);  // 기존 핸들러 재사용 가능
    break;
```

### Step 5: `ggml-hexagon.cpp` — Host 디스패치

```cpp
// supports_op에 추가
case GGML_OP_MUL_MAT:
    if (use_hmx_matmul(sess, op)) {
        supp = ggml_hexagon_supported_hmx_mul_mat(sess, op);
    } else {
        supp = ggml_hexagon_supported_mul_mat(sess, op);
    }
    break;

// graph_compute에 추가  
case GGML_OP_MUL_MAT:
    if (use_hmx_matmul(sess, node)) {
        ggml_hexagon_dispatch_op<init_hmx_matmul_req>(sess, node, flags);
    } else {
        // 기존 HVX 경로
    }
    break;
```

---

## 4. 파일 구조 요약 (수정/추가 대상)

```
ggml/src/ggml-hexagon/
├── ggml-hexagon.cpp          ← 수정: supports_op, graph_compute, init_*_req
├── htp/
│   ├── htp-msg.h             ← 수정: enum htp_op에 HMX Op 추가
│   ├── htp-ops.h             ← 수정: 커널 함수 선언 추가
│   ├── main.c                ← 수정: htp_packet_callback switch에 case 추가
│   ├── CMakeLists.txt        ← 수정: 새 소스 파일 추가  
│   ├── hmx-matmul-ops.c      ← 새 파일: HMX matmul 어댑터
│   └── hmx-flash-attn-ops.c  ← 새 파일: HMX flash attention 어댑터
├── CMakeLists.txt            ← 수정: htp-ops-lib include path 추가
│
htp-ops-lib/                   ← 커널 소스 참조 (직접 포함 or 라이브러리 링크)
├── include/dsp/ops.h          ← HMX 커널 함수 선언
├── src/dsp/ops/mat_mul.c      ← HMX matmul 구현
├── src/dsp/ops/flash_attn.c   ← HMX flash attention 구현
└── src/dsp/hmx_mgr.c          ← HMX 매니저 (충돌 방지 필요)
```

---

## 5. 주의사항 및 미해결 과제

1. **VTCM/HMX 매니저 충돌**: ggml-hexagon과 htp-ops-lib 각각 별도의 VTCM/HMX 매니저를 갖고 있음. 통합 시 하나로 통일하거나 어댑터 레이어 필요.

2. **가중치 permutation**: htp-ops-lib의 HMX matmul은 "permuted weight"를 요구. 기존 repack (Q4x4x2 등)과는 다른 포맷이므로 별도 repack 파이프라인 필요.

3. **아키텍처 감지**: v73 미만에서는 HMX 미지원이므로 런타임 아키텍처 체크 후 HVX fallback 경로 유지.

4. **데이터 타입 확장**: `enum htp_data_type`에 IQ4_NL 등 추가 필요시 htp-msg.h 확장.

5. **프로파일링 통합**: htp-ops-lib의 성능 측정과 ggml-hexagon의 `profile_data` 통합.

6. **테스트**: ggml-hexagon의 기존 테스트 프레임워크 (`tests/` 디렉토리)에 HMX Op 테스트 케이스 추가.

