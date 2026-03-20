# Host 측 (CPU) 디스패치 메커니즘 상세 분석

> 분석 파일: [ggml/src/ggml-hexagon/ggml-hexagon.cpp](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)

---

## 1. 세션 구조체 (`ggml_hexagon_session`)

**위치**: [ggml-hexagon.cpp L116~L146](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L116-L146)

```cpp
struct ggml_hexagon_session {
    ggml_backend_buffer_type buffer_type;
    ggml_backend_buffer_type repack_buffer_type;

    std::string      name;
    remote_handle64  handle;     // FastRPC 원격 핸들
    dspqueue_t       queue;     // DSP 큐 핸들
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    int              dev_id;
    std::atomic<int> op_pending; // 미완료 Op 카운터
    uint32_t         prof_usecs;
    uint32_t         prof_cycles;
    uint32_t         prof_pkts;
};
```

핵심 포인트:
- `dspqueue_t queue`: Op 요청/응답이 오가는 비동기 큐
- `op_pending`: 원자적 카운터로 미완료 Op 수를 추적
- 세션 당 하나의 `dspqueue`가 존재

---

## 2. Op 지원 판단 (`supports_op`)

**위치**: [ggml-hexagon.cpp L2984~L3095](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2984-L3095)

`ggml_backend_hexagon_device_supports_op()` 함수가 GGML 스케줄러에 의해 호출됨.

### 동작 방식:
1. **버퍼 호환성 확인** (`ggml_hexagon_supported_buffers`) — 모든 src/dst 텐서가 같은 세션의 Hexagon 버퍼에 있어야 함
2. **Op별 지원 함수 호출** — 각 Op마다 전용 `ggml_hexagon_supported_*()` 함수가 존재

### 지원 판단 함수 목록:

| 함수 | 위치 | 검사 내용 |
|------|------|-----------|
| `ggml_hexagon_supported_mul_mat` | [L1784](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1784) | src0 타입(Q4_0/Q8_0/MXFP4/F16), ne[0]%32==0, repack 버퍼 여부 |
| `ggml_hexagon_supported_mul_mat_id` | [L1835](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1835) | MoE용, src2 I32, src0 repack 필수 |
| `ggml_hexagon_supported_binary` | [L1863](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1863) | F32/F16 타입, contiguous tensors |
| `ggml_hexagon_supported_flash_attn_ext` | [L1752](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1752) | F16 KV, `opt_experimental` 플래그 필수 |
| `ggml_hexagon_supported_unary` | 각종 unary ops | F32 입출력 |
| `ggml_hexagon_supported_rope` | rope ops | F32/F16 |
| `ggml_hexagon_supported_ssm_conv` | SSM conv | F32?? contiguous 검사 |

### 지원하지 않을 경우 → **CPU fallback**
`supports_op()`이 `false`를 반환하면 ggml 스케줄러가 해당 Op을 CPU 백엔드로 자동 라우팅함.

---

## 3. 그래프 컴퓨트 (`graph_compute`)

**위치**: [ggml-hexagon.cpp L2552~L2672](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2552-L2672)

### `ggml_backend_hexagon_graph_compute()` 흐름:

```
for 각 node in graph:
    if !is_compute_op(node):
        continue

    flags 계산:
        - op_reuse_src1() → HTP_OPFLAGS_SKIP_QUANTIZE (재양자화 생략)
        - last op → HTP_OPFLAGS_EARLY_WAKEUP

    switch(node->op):
        case GGML_OP_MUL_MAT:
            ggml_hexagon_dispatch_op<init_binary_req<true/false>>()
        case GGML_OP_ROPE:
            ggml_hexagon_dispatch_op<init_rope_req>()
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_hexagon_dispatch_op<init_flash_attn_ext_req>()
        ...

sess->flush()  // 모든 미완료 Op 대기
```

### 주요 최적화:
- **VTCM 재사용**: 연속된 MUL_MAT op에서 같은 `src1`을 사용하면 `HTP_OPFLAGS_SKIP_QUANTIZE` 설정
- **그래프 리오더링**: [L2755~](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2755) 동일한 src1을 공유하는 MUL_MAT op을 연속 배치

---

## 4. Op 디스패치 템플릿 (`ggml_hexagon_dispatch_op`)

**위치**: [ggml-hexagon.cpp L2267~L2294](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2267-L2294)

```cpp
typedef size_t (*htp_req_init_func_t)(htp_general_req * req, dspqueue_buffer * bufs, const ggml_tensor * op);

template <htp_req_init_func_t _init_req_func>
static inline void ggml_hexagon_dispatch_op(ggml_hexagon_session *sess, 
                                             const struct ggml_tensor * op, 
                                             uint32_t flags) {
    htp_general_req req;
    memset(&req, 0, sizeof(req));
    req.flags = flags;  // opmask에 따라 SKIP_QUANTIZE/SKIP_COMPUTE 추가

    dspqueue_buffer bufs[HTP_MAX_PACKET_BUFFERS];
    size_t n_bufs = _init_req_func(&req, bufs, op);  // Op별 초기화 함수 호출

    sess->enqueue(req, bufs, n_bufs, opt_opsync);     // 큐에 전송
}
```

### 핵심 메커니즘:
- **C++ 템플릿으로 init 함수를 컴파일 타임 바인딩** → 가상 함수 오버헤드 제거
- 각 `init_*_req` 함수가:
  1. `req->op`에 `HTP_OP_*` 값 설정
  2. `req->op_params`에 ggml op_params 복사
  3. `htp_req_buff_init()`으로 각 src/dst 버퍼 참조 설정
  4. 사용한 버퍼 수(`n_bufs`) 반환

---

## 5. 버퍼 참조 초기화 (`htp_req_buff_init`)

**위치**: [ggml-hexagon.cpp L2224~L2260](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2224-L2260)

```cpp
static size_t htp_req_buff_init(htp_tensor *h, dspqueue_buffer * d, 
                                 const ggml_tensor * t, dspqbuf_type type) {
    // htp_tensor: DSP에서 사용할 텐서 메타데이터 (type, ne[], nb[])
    // dspqueue_buffer: fd, ptr, offset, size, flags (캐시 관리)
}
```

### 버퍼 타입별 캐시 관리:

| `dspqbuf_type` | 용도 | `dspqueue_buffer.flags` |
|-----------------|------|------------------------|
| `DSP_WRITE_CPU_READ` | DSP 출력 → CPU 읽기 | `FLUSH_SENDER` |
| `CPU_WRITE_DSP_READ` | CPU 입력 → DSP 읽기 | `FLUSH_SENDER \| INVALIDATE_RECIPIENT` |
| `CONSTANT` | 변경 없는 가중치 | `0` (캐시 관리 없음) |

---

## 6. 메시지 포맷 (`htp_general_req` / `htp_general_rsp`)

**위치**: [htp/htp-msg.h](../../ggml/src/ggml-hexagon/htp/htp-msg.h)

### 요청 구조체 (`htp_general_req`):
```cpp
struct htp_general_req {
    uint32_t op;                           // HTP_OP_* 열거형
    int32_t  op_params[16];                // ggml op_params (epsilon, mode 등)
    uint32_t flags;                        // HTP_OPFLAGS_*

    struct htp_tensor src0;                // 입력 텐서 0
    struct htp_tensor src1;                // 입력 텐서 1
    struct htp_tensor src2;                // 입력 텐서 2
    struct htp_tensor src3;                // 입력 텐서 3
    struct htp_tensor src4;                // 입력 텐서 4
    struct htp_tensor dst;                 // 출력 텐서
};
```

### `htp_tensor` 구조체:
```cpp
struct htp_tensor {
    uint32_t data;                // 버퍼 오프셋 (Host) / 데이터 포인터 (DSP)
    uint32_t type;                // htp_data_type 열거형
    uint32_t ne[4];               // 각 차원 요소 수
    uint32_t nb[4];               // 각 차원 stride (bytes)
};
```

### 응답 구조체 (`htp_general_rsp`):
```cpp
struct htp_general_rsp {
    uint32_t op;           // 어떤 Op에 대한 응답인지
    uint32_t status;       // HTP_STATUS_OK / ERR / ...
    uint32_t prof_usecs;   // 프로파일링: 마이크로초
    uint32_t prof_cycles;  // 프로파일링: 사이클
    uint32_t prof_pkts;    // 프로파일링: 인스트럭션 패킷
};
```

---

## 7. 큐 통신 흐름

**Enqueue** ([L147~L162](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L147-L162)):
```
sess->op_pending++
dspqueue_write(queue, flags=0, n_bufs, bufs, sizeof(req), &req, TIMEOUT)
// 비동기: 즉시 반환, DSP가 콜백에서 처리
```

**Flush** ([L165~L213](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L165-L213)):
```
while(op_pending > 0):
    dspqueue_read(queue, ..., &rsp, TIMEOUT)
    // rsp.status 검사
    // 프로파일링 데이터 기록
    op_pending--
```

---

## 8. init_*_req 함수별 버퍼 구성

| init 함수 | 위치 | 버퍼 수 | 구성 |
|-----------|------|---------|------|
| `init_binary_req<true>` | [L2296](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2296) | 3 | src0(CONSTANT) + src1(CPU→DSP) + dst(DSP→CPU) |
| `init_binary_req<false>` | [L2296](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2296) | 3 | src0(CPU→DSP) + src1(CPU→DSP) + dst(DSP→CPU) |
| `init_binary_id_req<>` | [L2363](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2363) | 4 | src0 + src1 + src2(ids) + dst |
| `init_unary_req` | [L2400](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2400) | 2~3 | src0 + [src1] + dst |
| `init_rope_req` | [L2481](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2481) | 3~4 | src0 + src1 + [src2] + dst |
| `init_flash_attn_ext_req` | [L2497](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2497) | 4~6 | src0(Q) + src1(K) + src2(V) + [src3(mask)] + [src4(sinks)] + dst |
| `init_set_rows_req` | [L2387](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2387) | 3 | src0 + src1 + dst |
| `init_get_rows_req` | [L2338](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2338) | 3 | src0 + src1 + dst |
| `init_cpy_req` | [L2329](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2329) | 2 | src0 + dst |
| `init_argsort_req` | [L2348](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2348) | 2 | src0 + dst |
| `init_sum_rows_req` | [L2471](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2471) | 2 | src0 + dst |
| `init_ssm_conv_req` | [L2515](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2515) | 3 | src0(CPU→DSP) + src1(CONSTANT) + dst |

---

## 9. Repack 메커니즘 (가중치 재배치)

**위치**: [ggml-hexagon.cpp L340~L700+](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L340)

양자화된 가중치(Q4_0, Q8_0, MXFP4)는 HVX에 최적화된 `*x4x2` 포맷으로 재배치됨:

- `repack_q4_0_q4x4x2()`: Q4_0 → Q4x4x2 (quants 먼저, scales 뒤에)
- `repack_q8_0_q8x4x2()`: Q8_0 → Q8x4x2
- `repack_mxfp4_mxfp4x4x2()`: MXFP4 → MXFP4x4x2

이 작업은 `repack_buffer_type`을 통해 **가중치 로딩 시 한 번만** 수행됨.

---

## 10. HMX Op 추가 시 Host 측 작업 요약

새로운 HMX 기반 Op (예: `htp-ops-lib`의 `hmx_mat_mul_permuted_w16a32`)을 추가하려면:

### 필수 수정 사항:

1. **`htp-msg.h`**: `enum htp_op`에 새 Op 코드 추가 (예: `HTP_OP_HMX_MUL_MAT`)
2. **`ggml-hexagon.cpp`**:
   - `ggml_hexagon_supported_*()` 함수 작성 또는 기존 함수에 분기 추가
   - `ggml_backend_hexagon_device_supports_op()`의 `switch`에 `case` 추가
   - `init_*_req()` 함수 작성 (버퍼 구성 정의)
   - `ggml_backend_hexagon_graph_compute()`의 `switch`에 디스패치 로직 추가
3. **DSP 측**: 별도 문서에서 분석

