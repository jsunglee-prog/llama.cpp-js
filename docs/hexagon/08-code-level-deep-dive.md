# 8. 코드 레벨 심층 분석 (Deep Dive)

> 이 문서는 기존 00~07 분석 문서를 보강하는 코드 레벨 상세 분석입니다.
> 각 파일의 핵심 함수, 라인 번호, 상수, 데이터 흐름 패턴을 기록합니다.

---

## 8.1 ggml-hexagon.cpp (3301줄) — CPU측 백엔드 전체 구현

> 위치: [ggml/src/ggml-hexagon/ggml-hexagon.cpp](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)

### 8.1.1 전역 옵션 변수 (라인 41~54)

```cpp
static size_t opt_ndev         = 1;     // DSP 디바이스 수
static size_t opt_nhvx         = 0;     // HVX 스레드 수 (0 = 모두 사용)
static int    opt_arch         = 0;     // 아키텍처 자동감지
static int    opt_etm          = 0;     // ETM 트레이싱
static int    opt_verbose      = 0;     // 상세 로그
static int    opt_profile      = 0;     // 프로파일링
static int    opt_hostbuf      = 1;     // 호스트 버퍼 ON (기본값)
static int    opt_experimental = 0;     // 실험적 기능 (Flash Attention)
static int    opt_opmask = HTP_OPMASK_QUEUE | HTP_OPMASK_QUANTIZE | HTP_OPMASK_COMPUTE;
static int    opt_opsync = 0;           // 동기 실행 모드
```

환경변수 매핑 (라인 ~3244, `ggml_hexagon_init()`):
| 환경변수 | 대응 옵션 | 기본값 |
|---------|----------|--------|
| `GGML_HEXAGON_EXPERIMENTAL` | `opt_experimental` | 0 |
| `GGML_HEXAGON_VERBOSE` | `opt_verbose` | 0 |
| `GGML_HEXAGON_HOSTBUF` | `opt_hostbuf` | 1 |
| `GGML_HEXAGON_OPMASK` | `opt_opmask` | 7 (모두 활성) |
| `GGML_HEXAGON_OPSYNC` | `opt_opsync` | 0 |
| `GGML_HEXAGON_PROFILE` | `opt_profile` | 0 |
| `GGML_HEXAGON_ETM` | `opt_etm` | 0 |
| `GGML_HEXAGON_NHVX` | `opt_nhvx` | 0 (전부) |
| `GGML_HEXAGON_NDEV` | `opt_ndev` | 1 |
| `GGML_HEXAGON_ARCH` | `opt_arch` | 0 (자동) |

### 8.1.2 세션 구조체 (`ggml_hexagon_session`, 라인 116~138)

```cpp
struct ggml_hexagon_session {
    ggml_backend_buffer_type buffer_type;        // 일반 버퍼 타입
    ggml_backend_buffer_type repack_buffer_type;  // Repack 버퍼 타입
    std::string      name;            // "hexagon0" 등
    remote_handle64  handle;          // DSP 핸들 (FastRPC)
    dspqueue_t       queue;           // dspqueue 핸들
    uint32_t         session_id;      // FastRPC 세션 ID
    uint32_t         domain_id;       // CDSP 도메인 ID
    uint64_t         queue_id;        // 큐 내보내기 ID
    int              dev_id;          // 디바이스 인덱스
    bool             valid_session/handle/queue/iface;
    std::atomic<int> op_pending;      // 미완료 Op 카운터
    uint32_t         prof_usecs/cycles/pkts;  // 프로파일링 누적
};
```

### 8.1.3 enqueue/flush 동기화 (라인 140~215)

**`enqueue()`** (라인 140):
- `op_pending++` (atomic inc)
- `dspqueue_write(queue, bufs, sizeof(req), ...)` → 비동기 전송
- `DSPQUEUE_TIMEOUT` 사용 (기본 5000ms)
- `sync=true`이면 즉시 `flush()` 호출

**`flush()`** (라인 163):
- `while(op_pending > 0)` 루프
- `dspqueue_read(queue, &rsp, ...)` → 블로킹 읽기
- 응답 크기 검증 (`rsp_size == sizeof(rsp)`)
- `AEE_EEXPIRED` 시 재시도 (타임아웃이지만 Op이 아직 실행 중)
- `rsp.status != HTP_STATUS_OK` 시 에러 로그 출력
- 프로파일링 데이터 수집
- `op_pending--` (atomic dec)

### 8.1.4 버퍼 관리 (라인 218~400)

**`ggml_backend_hexagon_buffer_context`** (라인 243):
- `rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG, size)`
  - 추가 4KB 패딩: `size += 4 * 1024`
- `rpcmem_to_fd(base)` → fd 획득
- `fastrpc_mmap(domain_id, fd, base, 0, size, FASTRPC_MAP_FD)` → DSP 매핑

**핵심 상수**:
- 버퍼 정렬: **128바이트** (HVX 벡터 크기와 동일)
- 최대 버퍼 크기: **1GB** (`1024 * 1024 * 1024`)

### 8.1.5 Repack 함수 (라인 433~600+)

#### Q4_0 → Q4x4x2

**`repack_row_q4x4x2()`** (라인 ~480):
1. QK_Q4_0x4x2 = 256 (8개 Q4_0 블록)
2. 4비트 quants를 먼저 연속 저장 (낮은 nibble + 높은 nibble 분리 후 재결합)
3. fp16 scales를 뒤에 8개씩 연속 저장
4. `dblk_size = 8 * 2` (16바이트, 8개 fp16 scale)
5. `qblk_size = qk / 2` (128바이트, 256개 4비트 값)

#### Q8_0 → Q8x4x2

**`repack_row_q8x4x2()`** (라인 ~759):
- 동일 원리: quants-first, scales-after
- 1바이트 quant이므로 qblk_size = 256바이트 = 정확히 2 HVX 벡터

#### MXFP4 → MXFP4x4x2

**`repack_row_mxfp4x4x2()`** (라인 ~1080):
- E8M0 스케일 형식 사용
- `kvalues_mxfp4[]` LUT로 4비트 → float 변환

### 8.1.6 set_tensor / get_tensor 라우팅 (라인 ~1402)

```cpp
ggml_backend_hexagon_buffer_set_tensor():
  switch(type):
    Q4_0  → repack_q4_0_q4x4x2()
    Q8_0  → repack_q8_0_q8x4x2()
    MXFP4 → repack_mxfp4_mxfp4x4x2()
    기타  → memcpy() (F32, F16 등)
```

**`get_tensor()`** (라인 ~1430): 역변환 수행
- `repack_q4x4x2_q4_0()`, `repack_q8x4x2_q8_0()` 등

### 8.1.7 세션 할당 (`allocate()`, 라인 ~1542)

1. `get_domain(3)` → CDSP 도메인
2. `remote_session_control(RESERVE_NEW_SESSION)` → 세션 예약
3. `remote_session_control(UNSIGNED_MODULE)` → unsigned PD
4. `htp_iface_open(uri, &handle)` → DSP 핸들 열기
   - URI: `"file:///libggml-htp-v{arch}.so?htp_iface_skel_handle_invoke"`
5. `dspqueue_create(domain_id, 128KB_req, 64KB_rsp)` → 큐 생성
6. `dspqueue_export(queue, &queue_id)` → 큐 ID
7. `htp_iface_start(handle, dev_id, queue_id, nhvx)` → DSP 시작

### 8.1.8 supports_op 상세 (라인 ~1798~2984)

#### `ggml_hexagon_supported_mul_mat()` (라인 ~1798)

| 조건 | 값 |
|------|---|
| src0 타입 | Q4_0, Q8_0, MXFP4, F16 |
| src0 ne[0] | `% 32 == 0` |
| src0 ne[1] | `≤ 16384` (VTCM 크기 제한) |
| src0 버퍼 | **repack buffer 필수** (양자화 타입) |
| src1 타입 | F32, F16 |
| src1 ne[2], ne[3] | `== 1` (2D만 지원, 4D 미지원 - F16은 예외) |
| dst 타입 | F32 |

#### `ggml_hexagon_supported_flash_attn_ext()` (라인 ~1850 부근)

- `opt_experimental` 필요
- Q: F16 또는 F32, K/V: F16, dst: F16 또는 F32
- optional F16 mask, optional F32 sinks

#### `ggml_hexagon_supported_argsort()`:
- F32 src, I32 dst, `ne[0] ≤ 16384`

#### 전체 supports_op switch (라인 ~2900):
```
항상 true: NONE, RESHAPE, VIEW, PERMUTE, TRANSPOSE
조건부: MUL_MAT, MUL_MAT_ID, MUL/ADD/SUB/DIV, ADD_ID,
        RMS_NORM, SCALE, SQR, SQRT, SUM_ROWS, SOFT_MAX,
        UNARY(SILU, GELU), GLU(SWIGLU, GEGLU),
        ROPE, FLASH_ATTN_EXT, SET_ROWS, GET_ROWS, CPY,
        ARGSORT, SSM_CONV
```

### 8.1.9 graph_compute() (라인 ~2569)

핵심 로직:
1. `last_compute_op()` 으로 마지막 연산 Op 인덱스 계산
2. 노드 순회: `is_compute_op(node)` 체크
3. **src1 재사용 최적화**: `op_reuse_src1()` → `HTP_OPFLAGS_SKIP_QUANTIZE`
4. **마지막 Op**: `HTP_OPFLAGS_EARLY_WAKEUP` 설정
5. Op별 `ggml_hexagon_dispatch_op<init_*_req>()` 호출
6. `sess->flush()` → 모든 응답 대기

### 8.1.10 graph_optimize() (라인 ~2780)

두 단계 최적화:
1. **Op 퓨전**: ADD/MUL/NORM/RMS_NORM 체인을 `ggml_can_fuse()`로 묶음 (MAX_FUSE=16)
2. **MUL_MAT 재정렬**: 같은 src1을 공유하는 MUL_MAT을 N_FORWARD=16 범위 내에서 연속 배치

### 8.1.11 디바이스/레지스트리 (라인 ~2960~3301)

- 디바이스 타입: `GGML_BACKEND_DEVICE_TYPE_GPU`
- 메모리 보고: **2GB** 고정
- async 지원: `true`
- host_buffer: `opt_hostbuf`
- 싱글톤 등록 (mutex 보호)
- `GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)` (마지막 줄)

---

## 8.2 htp-drv.cpp (~500줄) — FastRPC 드라이버 로더

> 위치: [ggml/src/ggml-hexagon/htp-drv.cpp](../../ggml/src/ggml-hexagon/htp-drv.cpp)

### 핵심 함수

| 함수 | 라인 | 역할 |
|------|------|------|
| `htpdrv_init()` | ~340 | `libcdsprpc.dll`/`.so` 로딩, 함수 포인터 바인딩 |
| `get_domain()` | ~420 | 도메인 ID로 도메인 구조체 조회 |
| `get_hex_arch_ver()` | ~430 | DSP 아키텍처 버전 쿼리 |
| `get_driver_path()` | (Windows) | Service Control Manager에서 qcnspmcdm 드라이버 경로 |

### 아키텍처 버전 매핑

```
capability byte → arch version:
  0x68 → v68
  0x69 → v69
  0x73 → v73
  0x75 → v75
  0x79 → v79
  0x81 → v81
```

`remote_handle_control(DSPRPC_GET_DSP_INFO)` + `ARCH_VER` 속성으로 조회.

### 바인딩되는 함수 목록

```
rpcmem_alloc2, rpcmem_free, rpcmem_to_fd
fastrpc_mmap, fastrpc_munmap
dspqueue_create, dspqueue_close, dspqueue_export
dspqueue_write, dspqueue_read
remote_handle64_open, remote_handle64_invoke, remote_handle64_close
remote_handle_control, remote_session_control
```

---

## 8.3 htp/main.c (1200줄) — DSP측 메인 루프

> 위치: [ggml/src/ggml-hexagon/htp/main.c](../../ggml/src/ggml-hexagon/htp/main.c)

### 8.3.1 전원 관리 (`htp_iface_open()`, 라인 ~28)

```c
HAP_power_set(NULL, &power_info):
  - core_corner = DCVS_VCORNER_MAX     // 코어 최대 클럭
  - bus_corner  = DCVS_VCORNER_MAX     // 버스 최대 클럭  
  - sleep_disable = 1                   // 절전 비활성화
  - hvx_power_on = 1                   // HVX 활성화
  - hmx_power_on = 1                   // HMX 활성화
  - client_class = HAP_DCVS_VCORNER_TURBO_PLUS_L1
```

### 8.3.2 VTCM 할당 (`vtcm_alloc()`, 라인 ~218)

```c
- HAP_query_vtcm_page() → vtcm_total(기본 8MB)
- HAP_compute_res_attr_set_vtcm_param(size, 1/*cached*/)
- HAP_compute_res_attr_set_hmx(1)
- HAP_compute_res_acquire(&attr, 100/*timeout_ms*/)
- vtcm_base = HAP_compute_res_attr_get_vtcm_ptr()
```

### 8.3.3 VTCM 경쟁 해소 (`vtcm_acquire()`, 라인 ~155)

**우선순위 범프 기법**:
1. 현재 우선순위 저장
2. 최고 우선순위로 임시 상향 → 다른 세션의 release 콜백 강제
3. Re-acquire
4. 원래 우선순위 복원

**Release 콜백** (`vtcm_release_callback()`, 라인 ~183):
- `vtcm_inuse == true`면 `vtcm_needs_release = true` (지연 해제)
- `vtcm_inuse == false`면 즉시 해제

### 8.3.4 서비스 초기화 (`htp_iface_start()`, 라인 ~250)

1. `dspqueue_import(queue_id, packet_callback)` → 큐 임포트
2. VTCM 할당
3. HW 스레드/HVX 유닛 탐지
4. DMA 큐 생성: 스레드당 1개, **용량 64 엔트리**
5. `worker_pool_init(n_hvx)` → 워커 풀 생성

### 8.3.5 메시지 루프 (`htp_packet_callback()`, 라인 ~1070)

```c
while(1):
  dspqueue_read_noblock() → 성공 시:
    req.op에 따라 디스패치:
      MUL_MAT       → proc_matmul_req()
      MUL_MAT_ID    → proc_matmul_id_req()
      MUL/ADD/SUB/DIV → proc_binary_req()
      RMS_NORM/SCALE → proc_unary_req()
      SOFTMAX/GLU   → proc_activations_req()
      ROPE          → proc_rope_req()
      FLASH_ATTN_EXT → proc_flash_attn_ext_req()
      SET_ROWS/GET_ROWS/CPY/ARGSORT/SUM_ROWS/SSM_CONV → proc_*_req()
    
  큐 비어있으면 → dspqueue_read() 블로킹 대기
```

각 `proc_*_req()` 내부 공통 흐름:
1. `htp_ops_context` 구성 (텐서 포인터 = dspqueue_buffer.ptr + offset)
2. `vtcm_acquire(ctx)` → VTCM 확보
3. `op_function(&octx)` → 실제 연산 실행
4. `vtcm_release(ctx)` → VTCM 해제
5. `send_htp_rsp(rsp)` → 응답 전송

**EARLY_WAKEUP**: `dspqueue_write_early_wakeup_noblock()` 호출 → CPU 조기 알림

---

## 8.4 htp/matmul-ops.c (2572줄) — HVX 행렬곱 커널

> 위치: [ggml/src/ggml-hexagon/htp/matmul-ops.c](../../ggml/src/ggml-hexagon/htp/matmul-ops.c)

### 8.4.1 핵심 상수

```c
#define MM_SPAD_SRC0_NROWS  16   // VTCM에 프리페치하는 src0 행 수
#define MM_SPAD_SRC1_NROWS  16   // 4D 경로의 src1 행 수
#define MM_SPAD_DST_NROWS    2   // dst 스크래치패드 행 수
#define QK_Q4_0x4x2        256   // Q4_0 슈퍼블록 크기
#define QK_Q8_0x4x2        256   // Q8_0 슈퍼블록 크기
#define QK_MXFP4x4x2       256   // MXFP4 슈퍼블록 크기
```

### 8.4.2 HVX 데이터 언패킹 함수

| 함수 | 역할 |
|------|------|
| `hvx_vec_load_q4x4x8()` | 4비트 언패킹: mask+shift, uint4→int4 (−8 바이어스) |
| `hvx_vec_load_q8x4x8()` | 8비트 직접 로드 (변환 불필요) |
| `hvx_vec_load_mxfp4x4x8()` | 4비트 언패킹 + LUT 변환 (`kvalues_mxfp4_lut`) |

### 8.4.3 내적 코어: `hvx_vec_rmpy_x8_full()`

```c
Q6_Vw_vrmpy_VbVb(a, b)  // 128-byte vector reduce-multiply
```

8개 HVX 벡터(8×128=1024바이트)에 대해:
1. `vrmpy` 8회 수행 → 8개 int32 부분합
2. `Q6_W_vdeal_VVR` + `Q6_Vw_vadd_VwVw` 리덕션 트리
3. 최종 32개 FP32 부분합 → 하나의 HVX 벡터

### 8.4.4 vec_dot 변형 (1x1, 2x1, 2x2)

**Q4_0/Q8_0/MXFP4 공통 흐름**:
```
1. 양자값 로드 → hvx_vec_load_*x4x8()
2. vrmpy 누적 → hvx_vec_rmpy_x8_full/nloe()
3. 스케일 로드 (fp16 or e8m0)
4. 스케일 결합 (qf32 multiply)
5. 스케일 × 누적기
6. 행 합계 리덕션 → hvx_vec_reduce_sum_f32()
7. 스토어
```

**2x2 변형**: 4개 출력 동시 계산 (src0 2행 × src1 2열)
- src0 행을 재사용 (2열에 대해)
- src1 열을 재사용 (2행에 대해)

### 8.4.5 MXFP4 E8M0 스케일 변환

```c
// 32개 uint8 E8M0 → 32개 FP32
HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
r0_d = Q6_V_vdelta_VV(r0_d, expand);     // uint8→uint32 확장
r0_d = Q6_V_vand_VV(r0_d, 0x000000ff);   // 마스킹
r0_d = Q6_Vw_vasl_VwR(r0_d, 23);         // <<23 → FP32 지수 위치
```

`expand_x32_e8m0[128]`: vdelta 제어 벡터 (바이트 셔플 패턴)

### 8.4.6 타일링 전략

#### `matmul_2d()` (라인 ~1569) — 주 경로

**전제**: src1(입력)이 VTCM에 전체 로딩됨

```
VTCM 레이아웃:
  src0_spad: 16행 × row_size_padded × n_threads (스레드별)
  src1_spad: 전체 src1 (공유)
  dst_spad:  2행 × dst_row_size × n_threads (스레드별)
```

**DMA 파이프라이닝 패턴**:
1. 초기 프리페치: src0의 첫 16행을 VTCM으로 DMA 전송
2. 메인 루프:
   - `dma_queue_pop()` → 현재 행 데이터 확보
   - src1 열 순회: `vec_dot_2x2()` (2행×2열 타일)
   - `dma_queue_push_ddr_to_vtcm()` → 다음 16행 비동기 프리페치
3. 마지막 홀수 행: `vec_dot_1x1()` 폴백

#### `matmul_4d()` (라인 ~1500) — F16 폴백 경로

- 4D 브로드캐스팅 지원 (ne02>1 또는 ne03>1)
- VTCM 타일링 없이 DRAM 직접 접근
- 블록 타일링: blck_0=64, blck_1=64
- `fastdiv` 유틸리티로 정수 나눗셈 최적화

#### `matvec_2d()` (라인 ~1698) — 벡터-행렬곱

- src1_nrows == 1일 때 선택
- dst를 VTCM 임시 버퍼에 계산 후 DMA로 DRAM에 복사

### 8.4.7 동적 양자화 (라인 ~2000)

**`quantize_row_f32_q8x4x2()`**:
1. F32 입력 → HVX로 abs max 계산
2. scale = max / 127.0
3. F32→FP16 변환 → scale 역수 곱셈 → int8 변환 (`Q6_Vb_vpack_VhVh_sat`)
4. quants-first, scales-after 레이아웃으로 저장

**`FP32_QUANTIZE_GROUP_SIZE`**: 32, 64, 128 중 선택 (컴파일타임)

**`quantize_f32_f16()`**: F32→F16 변환 (F16-F16 matmul 경로용)

### 8.4.8 `op_matmul()` — 최상위 진입점 (라인 ~2380)

경로 선택 로직:
```
src0이 F16일 때:
  ├── src1이 VTCM에 들어가면 → F16-F16 최적화 경로 (matmul_2d)
  │     src1_type이 F32면: quantize_f32_f16() → F16 변환
  │     src1_type이 F16면: quantize_f16_f16() → 단순 복사
  │
  └── src1이 VTCM에 안 들어가면 → F16-F32/F16 DDR 폴백 (matmul_4d)
       4D 브로드캐스팅 / permuted 텐서 지원

src0이 Q4_0/Q8_0/MXFP4일 때:
  → quantize_f32_q8x4x2() + matmul_2d/matvec_2d
```

---

## 8.5 htp/flash-attn-ops.c (714줄) — Flash Attention 커널

> 위치: [ggml/src/ggml-hexagon/htp/flash-attn-ops.c](../../ggml/src/ggml-hexagon/htp/flash-attn-ops.c)

### 핵심 상수

```c
#define FLASH_ATTN_BLOCK_SIZE  (32*2)  // 64행 단위 K/V 블록
```

### 8.5.1 HVX F16 내적 함수

| 함수 | 설명 |
|------|------|
| `hvx_dot_f16_f16_aa()` | 단일 F16×F16 내적 → F32 |
| `hvx_dot_f16_f16_aa_rx4()` | 4행 배치 내적 (y 재사용) |
| `hvx_dot_f16_f16_aa_rx32()` | 32행 배치 (rx4를 8회) |
| `hvx_mad_f32_f16_aa()` | F32 += F16 × F16_scalar (V 누적) |
| `hvx_mad_f32_f16_aa_rx2()` | 2-source MAD 변형 |

### 8.5.2 `flash_attn_ext_f16_thread()` — 스레드별 처리

VTCM 레이아웃 (스레드별):
```
spad_q: Q 1행 (DK × fp16, 128B 정렬)
spad_k: K 블록 ×2 (double buffer)
spad_v: V 블록 ×2 (double buffer)
spad_m: Mask 블록 ×2 (있을 경우)
spad_a: VKQ32 누적기 (DV × FP32)
```

**Online Softmax 알고리즘**:
```
for each K/V block:
  1. QK^T 계산: scores = Q · K[block]  (hvx_dot_f16_f16_aa_rx32)
  2. Softcap: scores = tanh(scores) * logit_cap  (옵션)
  3. Mask 적용: scores += slope * mask[block]
  4. max 업데이트: M_new = max(M_old, block_max)
  5. 스케일: VKQ *= exp(M_old - M_new)
  6. P 계산: P = exp(scores - M_new)
  7. VKQ 누적: VKQ += P × V[block]  (hvx_mad_f32_f16_aa_rx2)
  8. S 업데이트: S = S * exp(M_old - M_new) + sum(P)

최종: result = VKQ / S
```

**DMA 더블 버퍼링**:
- 처음 2개 블록 프리페치
- 블록 i 처리 중 → 블록 i+2 프리페치 (`ib + 2 < n_blocks`)
- `dma_queue_pop()` → K, V, Mask 순서로 대기

### 8.5.3 `op_flash_attn_ext()` — 진입점

VTCM 필요량 계산:
```
total_spad = (Q_per_thread + K_block×2 + V_block×2 + M_block×2 + VKQ_acc) × n_threads
```

`vtcm_size < total_spad`이면 `HTP_STATUS_VTCM_TOO_SMALL` 반환.

---

## 8.6 htp/softmax-ops.c (420줄)

> 위치: [ggml/src/ggml-hexagon/htp/softmax-ops.c](../../ggml/src/ggml-hexagon/htp/softmax-ops.c)

### 최적화 경로 선택

```c
if (aligned && row_stride_aligned)  → opt_path = 1
  → hvx_fast_softmax_prep_f32() + hvx_fast_softmax_f32()  // 빠른 경로

else  → opt_path = 0
  → hvx_scale_f32() + 수동 mask 적용
  → hvx_reduce_max_f32() → hvx_softmax_f32() → hvx_scale_f32()  // 느린 경로
```

**ALiBi 지원**: `max_bias > 0`이면 헤드별 slope 계산:
```c
slope = h < n_head_log2 ? pow(m0, h+1) : pow(m1, 2*(h-n_head_log2)+1)
```

VTCM 사용: 스레드별 3행 (src0, src1, dst 스크래치패드)

---

## 8.7 htp/worker-pool.c (294줄) — QuRT 스레드 풀

> 위치: [ggml/src/ggml-hexagon/htp/worker-pool.c](../../ggml/src/ggml-hexagon/htp/worker-pool.c)

### 핵심 상수

```c
#define MAX_NUM_WORKERS          10
#define WORKER_THREAD_STACK_SZ   (2*16384)  // 32KB per thread
```

### 작업 분배 메커니즘

```c
worker_pool_run_jobs(context, jobs[], n):
  1. 작업 배열 복사
  2. atomic_store(&next_job, 1)     // 워커는 1번부터
  3. n_pending = n - 1
  4. seqn++ + qurt_futex_wake()     // 워커 깨우기
  5. 메인 스레드가 job[0] 직접 실행
  6. while(n_pending > 0) 스핀 대기
```

**워커 메인 루프** (`worker_pool_main()`):
```c
while(1):
  qurt_futex_wait(&seqn, my_seqn)   // 시퀀스 번호 변경 대기
  while(true):
    i = atomic_fetch_add(&next_job, 1)  // 다음 작업 원자적 인출
    if (i >= n_jobs) break
    job[i].func(n_jobs, i, job[i].data)  // 실행
    atomic_fetch_sub(&n_pending, 1)      // 완료 신호
```

### `worker_pool_run_func()` — 단순 인터페이스

모든 워커에게 같은 함수+데이터를 전달 (인덱스만 다름):
```c
func(n_threads, thread_id, shared_data)
```

---

## 8.8 htp/htp-msg.h (156줄) — 메시지 프로토콜

> 위치: [ggml/src/ggml-hexagon/htp/htp-msg.h](../../ggml/src/ggml-hexagon/htp/htp-msg.h)

### Op 열거형 (`htp_op`)

```c
HTP_OP_MUL=0, ADD=1, SUB=2, DIV=3,
MUL_MAT, MUL_MAT_ID, RMS_NORM,
UNARY_SILU, UNARY_GELU,
GLU_SWIGLU, GLU_SWIGLU_OAI, GLU_GEGLU,
SOFTMAX, ADD_ID, ROPE, FLASH_ATTN_EXT,
SET_ROWS, GET_ROWS, SCALE, CPY,
ARGSORT, SQR, SQRT, SUM_ROWS, SSM_CONV
```

### 요청/응답 구조체

**`htp_general_req`** (~160바이트):
- `op` (4B), `op_params[64]` (64B), `flags` (4B)
- `src0~src4` + `dst` (각 `htp_tensor`: 40B = data+type+ne[4]+nb[4])

**`htp_general_rsp`** (64바이트 고정):
- `op` (4B), `status` (4B)
- `prof_usecs` (4B), `prof_cycles` (4B), `prof_pkts` (4B)
- `unused[44]` (패딩)

### 내부 양자화 포맷 상수

```c
#define QK_Q4_0x4x2   256
#define QK_Q8_0x4x2   256
#define QK_MXFP4x4x2  256
#define HTP_MAX_DIMS   4
#define HTP_MAX_OP_PARAMS  64
```

---

## 8.9 핵심 자료구조 헤더 요약

### htp-ctx.h (35줄)

```c
struct htp_context {
    dspqueue_t    queue;
    dma_queue *   dma[HTP_MAX_NTHREADS];  // 스레드별 DMA 큐 (max 10)
    worker_pool_context_t worker_pool;
    uint32_t      n_threads;
    qurt_thread_t thread_id;
    unsigned int  prio;
    uint8_t *     vtcm_base;
    size_t        vtcm_size;
    void *        vtcm_rctx;              // compute resource 핸들
    atomic_int    vtcm_valid;
    atomic_int    vtcm_inuse;
    atomic_int    vtcm_needs_release;
    int           opmask;
};
```

### htp-ops.h (69줄)

```c
struct htp_spad {
    uint8_t * data;
    size_t    stride;
    size_t    size;
    size_t    size_per_thread;
};

struct htp_ops_context {
    struct htp_context * ctx;
    enum htp_op          op;
    int32_t              op_params[HTP_MAX_OP_PARAMS/sizeof(int32_t)];
    struct htp_tensor    src0, src1, src2, src3, src4, dst;
    struct htp_spad      src0_spad, src1_spad, src2_spad, src3_spad, dst_spad;
    worker_pool_context_t worker_pool;
    uint32_t             n_threads;
    uint32_t             flags;
};
```

### hvx-types.h (39줄)

```c
#define VLEN       128     // HVX 벡터 길이 (바이트)
#define VLEN_FP32   32     // FP32 원소 수 per vector
#define VLEN_FP16   64     // FP16 원소 수 per vector

typedef union {
    HVX_Vector v;
    uint8_t    b[128];
    uint16_t   h[64];
    uint32_t   w[32];
    __fp16     fp16[64];
    float      fp32[32];
} HVX_VectorAlias;
```

---

## 8.10 ggml-backend.cpp의 그래프 분할 상세 (라인 925~1300)

> 위치: [ggml/src/ggml-backend.cpp](../../ggml/src/ggml-backend.cpp)

### 패스 5 (라인 ~1155) — 실제 Split 생성

핵심 로직:
```
for each node:
  if (node_backend != current_split_backend || need_new_split):
    → 새 split 생성
    → split.backend_id = node_backend
    → split.i_start = i

  for each src of node:
    if (src_backend != current_backend && !buffer_supported):
      → 복사 텐서 생성: ggml_dup_tensor_layout()
      → split.inputs[] 에 추가
      → node->src[j] 를 복사본으로 교체
```

**`need_new_split` 발생 조건**:
1. 가중치가 비호환 백엔드에 있을 때 (메모리 재사용 위해 분할)
2. split 입력이 `GGML_SCHED_MAX_SPLIT_INPUTS` 한도에 도달

### Hexagon과의 상호작용

- Hexagon repack 버퍼는 `is_host=false` → 스케줄러가 이 버퍼를 CPU가 읽을 수 없다고 판단
- Hexagon이 `supports_op() = true`인 Op → 패스 2에서 Hexagon으로 확장
- Hexagon이 `supports_op() = false`인 Op → CPU로 폴백, split 경계 생성

---

## 8.11 dspqueue 버퍼 타입과 캐시 관리

### 버퍼 타입 (dispatch 시 사용)

| 타입 | 의미 | 사용처 |
|------|------|--------|
| `DSPQBUF_TYPE_CONSTANT` | CPU→DSP 읽기전용 (캐시 가능) | 모델 가중치 (src0) |
| `DSPQBUF_TYPE_CPU_WRITE_DSP_READ` | CPU 쓰기 → DSP 읽기 | 입력 텐서 (src1~src4) |
| `DSPQBUF_TYPE_DSP_WRITE_CPU_READ` | DSP 쓰기 → CPU 읽기 | 출력 텐서 (dst) |

**CONSTANT 타입의 의미**: `init_binary_req<true>` 템플릿에서:
```cpp
template<bool src0_constant>
static inline size_t init_binary_req(...) {
    n_bufs += htp_req_buff_init(&req->src0, &bufs[n_bufs], src0,
        src0_constant ? DSPQBUF_TYPE_CONSTANT : DSPQBUF_TYPE_CPU_WRITE_DSP_READ);
}
```

양자화된 가중치(src0)는 `CONSTANT`로 마킹 → DSP가 캐시에 유지 가능 → 반복 접근 성능 향상.

### 응답 시 플러시 플래그

DSP에서 응답 전송 시:
```c
FLUSH_SENDER | INVALIDATE_RECIPIENT
```
- `FLUSH_SENDER`: DSP의 L2 캐시를 DRAM으로 플러시
- `INVALIDATE_RECIPIENT`: CPU의 캐시를 무효화하여 최신 DRAM 데이터 읽도록 보장
