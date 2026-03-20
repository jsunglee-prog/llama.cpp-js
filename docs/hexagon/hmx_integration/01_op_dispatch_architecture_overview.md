# ggml-hexagon Op 디스패치 아키텍처 분석

> 분석 대상: `ggml/src/ggml-hexagon/` 디렉토리 전체  
> 목적: HMX 기반 `htp-ops-lib` 연산 추가를 위한 기존 아키텍처 파악

## 1. 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOST (CPU / HLOS)                            │
│                                                                 │
│  ggml_backend_hexagon_graph_compute()                          │
│       │                                                         │
│       ├─ supports_op() → ggml_hexagon_supported_*()            │
│       │                                                         │
│       ├─ ggml_hexagon_dispatch_op<init_*_req>()                │
│       │       │                                                 │
│       │       ├─ init_*_req() → htp_general_req + dspqueue_buf │
│       │       └─ sess->enqueue() → dspqueue_write()            │
│       │                                                         │
│       └─ sess->flush() → dspqueue_read() ← htp_general_rsp    │
│                                                                 │
├────────────── FastRPC / dspqueue ───────────────────────────────┤
│                                                                 │
│                    DSP (HTP / skel)                             │
│                                                                 │
│  htp_packet_callback()                                         │
│       │                                                         │
│       ├─ dspqueue_read_noblock() → htp_general_req             │
│       │                                                         │
│       ├─ switch(req.op) → proc_*_req()                         │
│       │       │                                                 │
│       │       ├─ htp_ops_context 구성                          │
│       │       ├─ vtcm_acquire()                                │
│       │       ├─ op_*() 호출 (HVX 커널)                       │
│       │       ├─ vtcm_release()                                │
│       │       └─ send_htp_rsp()                                │
│       │                                                         │
│       └─ dspqueue_write() → htp_general_rsp                    │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 통신 인터페이스

- **IDL 파일**: [htp/htp_iface.idl](../../ggml/src/ggml-hexagon/htp/htp_iface.idl) — 세션 `start`/`stop` 및 ETM 제어만 정의
- **실제 Op 디스패치**: `dspqueue` (FastRPC 기반 비동기 큐) 를 통해 수행됨. IDL은 세션 관리만 담당
- **메시지 포맷**: `htp_general_req` (요청) / `htp_general_rsp` (응답) 구조체가 `dspqueue_write`/`dspqueue_read`로 교환됨

## 3. 지원 Op 목록

[htp-msg.h](../../ggml/src/ggml-hexagon/htp/htp-msg.h)의 `enum htp_op`:

| HTP Op | GGML Op | DSP 핸들러 | 커널 |
|--------|---------|------------|------|
| `HTP_OP_MUL` | `GGML_OP_MUL` | `proc_binary_req` | `op_binary` |
| `HTP_OP_ADD` | `GGML_OP_ADD` | `proc_binary_req` | `op_binary` |
| `HTP_OP_SUB` | `GGML_OP_SUB` | `proc_binary_req` | `op_binary` |
| `HTP_OP_DIV` | `GGML_OP_DIV` | `proc_binary_req` | `op_binary` |
| `HTP_OP_MUL_MAT` | `GGML_OP_MUL_MAT` | `proc_matmul_req` | `op_matmul` |
| `HTP_OP_MUL_MAT_ID` | `GGML_OP_MUL_MAT_ID` | `proc_matmul_id_req` | `op_matmul_id` |
| `HTP_OP_RMS_NORM` | `GGML_OP_RMS_NORM` | `proc_unary_req` | `op_unary` |
| `HTP_OP_UNARY_SILU` | `GGML_OP_UNARY(SILU)` | `proc_activations_req` | `op_activations` |
| `HTP_OP_UNARY_GELU` | `GGML_OP_UNARY(GELU)` | `proc_activations_req` | `op_activations` |
| `HTP_OP_GLU_SWIGLU` | `GGML_OP_GLU(SWIGLU)` | `proc_activations_req` | `op_activations` |
| `HTP_OP_GLU_SWIGLU_OAI` | `GGML_OP_GLU(SWIGLU_OAI)` | `proc_activations_req` | `op_activations` |
| `HTP_OP_GLU_GEGLU` | `GGML_OP_GLU(GEGLU)` | `proc_activations_req` | `op_activations` |
| `HTP_OP_SOFTMAX` | `GGML_OP_SOFT_MAX` | `proc_activations_req` | `op_softmax` |
| `HTP_OP_ADD_ID` | `GGML_OP_ADD_ID` | `proc_add_id_req` | `op_binary` |
| `HTP_OP_ROPE` | `GGML_OP_ROPE` | `proc_rope_req` | `op_rope` |
| `HTP_OP_FLASH_ATTN_EXT` | `GGML_OP_FLASH_ATTN_EXT` | `proc_flash_attn_ext_req` | `op_flash_attn_ext` |
| `HTP_OP_SET_ROWS` | `GGML_OP_SET_ROWS` | `proc_set_rows_req` | `op_set_rows` |
| `HTP_OP_GET_ROWS` | `GGML_OP_GET_ROWS` | `proc_get_rows_req` | `op_get_rows` |
| `HTP_OP_SCALE` | `GGML_OP_SCALE` | `proc_unary_req` | `op_unary` |
| `HTP_OP_CPY` | `GGML_OP_CPY` | `proc_cpy_req` | `op_cpy` |
| `HTP_OP_ARGSORT` | `GGML_OP_ARGSORT` | `proc_argsort_req` | `op_argsort` |
| `HTP_OP_SQR` | `GGML_OP_SQR` | `proc_unary_req` | `op_unary` |
| `HTP_OP_SQRT` | `GGML_OP_SQRT` | `proc_unary_req` | `op_unary` |
| `HTP_OP_SUM_ROWS` | `GGML_OP_SUM_ROWS` | `proc_sum_rows_req` | `op_sum_rows` |
| `HTP_OP_SSM_CONV` | `GGML_OP_SSM_CONV` | `proc_ssm_conv_req` | `op_ssm_conv` |

