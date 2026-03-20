# Hexagon Backend `ggml-hexagon.cpp` 분석 Q&A

> 날짜: 2026-03-21
> 대상 파일: `ggml/src/ggml-hexagon/ggml-hexagon.cpp` (3301줄)
> 관련 디렉토리: `ggml/src/ggml-hexagon/htp/`

---

## 목차

1. [Q1. ggml_backend_hexagon_graph_compute 함수 분석 및 RPC 여부](#q1)
2. [Q2. ggml-hexagon.cpp 파일 전체 구조 분석](#q2)
3. [Q3. dspqueue 오버헤드 예상치](#q3)
4. [Q4. 비동기 큐 처리에서 그래프 최적화가 필요한 이유](#q4)

---

<a id="q1"></a>
## Q1. `ggml_backend_hexagon_graph_compute` 함수 분석 및 RPC 여부

### 질문

> `ggml_backend_hexagon_graph_compute` 함수를 분석해줘.
> 실제로 그래프가 생성되면, 그래프의 각 노드를 돌면서, DSP에 명령하면서 수행하는 것인지?
> 그러하다면 모든 노드에서 CPU-HTP간에 RPC가 이루어지는 것인지?

### 답변

#### 전체 실행 흐름

**핵심: 그래프의 각 노드를 순회하며 DSP에 명령을 보낸다. 단, 전통적인 FastRPC가 아니라 `dspqueue`(공유 메모리 기반 메시지 큐)를 사용한다.**

```
CPU측 graph_compute  ──for 루프──>  dispatch_op (노드별)  ──dspqueue_write──>  [공유메모리 큐]
                                                                                    │
                                                                                    ▼
                                                            HTP측 htp_packet_callback
                                                            (main.c에서 dspqueue_read_noblock)
                                                                    │
                                                                    ▼
                                                            proc_matmul_req / proc_unary_req 등
                                                                    │
                                                                    ▼
                                                            send_htp_rsp ──dspqueue_write──> [공유메모리 큐]
                                                                                                    │
flush() <──dspqueue_read──────────────────────────────────────────────────────────────────────────────┘
```

#### CPU측 실행 단계 (노드별)

위치: `ggml-hexagon.cpp` L2552-L2668

1. **`last_compute_op(graph)`** (L2557) — 마지막 연산 노드의 인덱스를 미리 찾음 (L2541-L2549)
2. **for 루프로 모든 노드 순회** — `is_compute_op()`으로 비연산 노드는 스킵 (L2561)
3. **플래그 설정**:
   - `HTP_OPFLAGS_SKIP_QUANTIZE`: 이전 Op과 동일한 src1을 재사용하면 양자화 생략 (L2570-L2572)
   - `HTP_OPFLAGS_EARLY_WAKEUP`: 마지막 Op에만 설정하여 조기 응답 통지 요청 (L2578-L2579)
4. **Op 타입별 dispatch** — `ggml_hexagon_dispatch_op<init_xxx_req>()` 템플릿 호출 (L2582-L2662)
5. **`sess->flush()`** — 모든 Op이 큐에 들어간 후, 미완료 응답을 모두 수신할 때까지 대기 (L2668)

#### `ggml_hexagon_dispatch_op` 상세 (L2267-L2292)

```cpp
// 1) htp_general_req 구조체 생성 (연산 종류, 텐서 정보, 플래그)
// 2) _init_req_func()로 버퍼 참조(fd, offset, size) 설정
// 3) sess->enqueue()로 dspqueue에 write
```

- `_init_req_func`는 템플릿 파라미터로 `init_binary_req`, `init_unary_req` 등이 전달됨
- `htp_req_buff_init` (L2224)에서 **데이터를 복사하지 않고**, 공유 메모리의 `fd`와 `offset`만 전달

#### 통신 메커니즘: dspqueue (공유 메모리 큐) — 전통적 RPC가 아님

| 구분 | 전통적 FastRPC | 이 코드의 dspqueue |
|------|---------------|-------------------|
| **메커니즘** | 커널을 통한 함수 호출 | 공유 메모리 링 버퍼 |
| **호출 비용** | 높음 (syscall + context switch) | 낮음 (메모리 write + 시그널) |
| **데이터 전달** | 복사 또는 매핑 | **이미 매핑된 공유 메모리 참조** (fd + offset) |
| **비동기 지원** | 기본적으로 동기 | **비동기** (write 후 즉시 반환) |

**핵심 포인트:**

- **`dspqueue_write()`** (L145): CPU가 요청 메시지를 큐에 넣음 → **즉시 반환** (블로킹 아님, `opt_opsync=0`일 때)
- HTP측 **`htp_packet_callback()`** (`htp/main.c` L1008): DSP가 콜백으로 큐에서 메시지를 읽고 처리
- **`flush()`** (L163-L216): 마지막에 `dspqueue_read()`로 모든 응답을 수신할 때까지 대기

#### 노드별 통신 발생 여부

**네, 노드별로 `dspqueue_write`가 1회씩 발생한다.** 하지만:

- **비동기 파이프라인**: 기본 설정(`opt_opsync=0`)에서는 CPU가 모든 노드의 요청을 **연속으로 큐에 투입**하고, DSP는 그걸 **콜백에서 연속으로 처리**한다. CPU와 DSP가 **병렬 동작**한다.
- **데이터 복사 없음**: 버퍼는 `rpcmem_alloc`으로 할당한 공유 메모리이며, `fastrpc_mmap`으로 이미 매핑되어 있어서 `dspqueue_write`에는 메타데이터(fd, offset, size)만 전달된다.
- **동기 모드**: `opt_opsync=1`로 설정하면 각 노드마다 `flush()`를 호출하여 응답을 기다리는 동기 방식으로 전환된다.

#### graph_compute 내 최적화 기법

| 최적화 | 설명 | 위치 |
|--------|------|------|
| **양자화 스킵** | 연속 Op이 동일한 src1을 사용하면 재양자화 생략 | L2570-L2572 |
| **Early Wakeup** | 마지막 Op에만 조기 통지 요청 → CPU가 더 빨리 깨어남 | L2578-L2579 |
| **캐시 관리** | 버퍼별로 FLUSH/INVALIDATE 플래그를 세밀하게 제어 | `htp_req_buff_init` L2246-L2259 |
| **Constant 버퍼** | weights(src0)는 `DSPQBUF_TYPE_CONSTANT`로 캐시 관리 생략 | L2323 |

#### 요약

> **모든 노드에서 "전통적인 FastRPC"가 발생하는 것은 아니다.** `dspqueue`라는 공유 메모리 기반 메시지 큐를 사용하여, 노드별로 경량 메시지(~수백 바이트)를 비동기로 큐에 넣고, DSP측에서 콜백으로 연속 처리한다. 텐서 데이터는 이미 매핑된 공유 메모리를 통해 zero-copy로 접근하므로, 노드별 통신 오버헤드는 전통적 RPC 대비 매우 작다.

---

<a id="q2"></a>
## Q2. `ggml-hexagon.cpp` 파일 전체 구조 분석

### 질문

> `ggml-hexagon.cpp` 파일도 분석해줘.

### 답변

#### 파일 전체 구조 맵

| 라인 범위 | 섹션 | 설명 |
|-----------|------|------|
| L1-38 | **헤더/인클루드** | 시스템 헤더, Hexagon SDK 헤더 (`dspqueue.h`, `rpcmem.h`), ggml 내부 헤더 |
| L40-51 | **전역 옵션** | 환경변수 기반 런타임 설정 |
| L56-78 | **유틸리티** | 정렬, 상태 문자열 변환 |
| L83-108 | **디버그 헬퍼** | Op 실행/지원/프로파일 덤프 |
| L111-216 | **세션 관리** | `ggml_hexagon_session` 구조체, `enqueue()`, `flush()` |
| L219-306 | **버퍼 컨텍스트** | `rpcmem_alloc2` → `fastrpc_mmap` 공유메모리 할당 |
| L307-1450 | **버퍼 인터페이스 + Repack** | 텐서 set/get, Q4/Q8/MXFP4 repack 로직 |
| L1451-1534 | **버퍼 타입 인터페이스** | alignment(128), max_size(1GB), is_host |
| L1535-1740 | **세션 할당/해제** | FastRPC 세션, dspqueue 생성, HTP skel 로드 |
| L1741-2190 | **Op 지원 검사** | `ggml_hexagon_supported_*` 함수들 (15개) |
| L2193-2266 | **DSP 버퍼 초기화** | `htp_req_tensor_init`, `htp_req_buff_init` |
| L2267-2292 | **Op Dispatch** | `ggml_hexagon_dispatch_op` 템플릿 |
| L2295-2520 | **요청 초기화 함수들** | `init_binary_req`, `init_unary_req` 등 (12개) |
| L2525-2668 | **Graph Compute** | `ggml_backend_hexagon_graph_compute` |
| L2681-2846 | **Graph Optimize** | 노드 퓨전 + 리오더링 (VTCM 재사용 최적화) |
| L2847-3100 | **Backend/Device 인터페이스** | `supports_op`, `supports_buft`, device props |
| L3100-3301 | **Registry** | 환경변수 파싱, `htpdrv_init`, 세션 생성 |

#### 전역 옵션 (L40-51)

```
GGML_HEXAGON_NDEV     → opt_ndev (기본 1, 최대 16)     : HTP 세션 수
GGML_HEXAGON_NHVX     → opt_nhvx (기본 0=전부)         : HVX 스레드 수
GGML_HEXAGON_ARCH     → opt_arch (자동감지)             : v73/v75
GGML_HEXAGON_OPMASK   → opt_opmask (기본 0x7)          : 큐잉/양자화/계산 on/off
GGML_HEXAGON_OPSYNC   → opt_opsync (기본 0)             : 동기 모드
GGML_HEXAGON_HOSTBUF  → opt_hostbuf (기본 1)            : host 버퍼 활성화
GGML_HEXAGON_PROFILE  → opt_profile                     : 프로파일링
GGML_HEXAGON_VERBOSE  → opt_verbose                     : 디버그 로그
```

#### 세션 관리 (`ggml_hexagon_session`, L111-216)

세션은 CPU ↔ HTP 통신의 핵심 단위.

**멤버 변수:**

| 필드 | 설명 |
|------|------|
| `handle` | FastRPC 원격 핸들 (HTP skel 로드용) |
| `queue` | `dspqueue_t` — 공유 메모리 메시지 큐 |
| `session_id` | FastRPC 세션 ID |
| `domain_id` | CDSP 도메인 ID (기본 3) |
| `op_pending` | 미완료 Op 카운터 (atomic) |

**`enqueue()`** (L141-L159):
- `dspqueue_write()`로 요청 메시지 + 버퍼 참조를 큐에 삽입
- `opt_opsync=true`이면 즉시 `flush()` 호출

**`flush()`** (L164-L216):
- `op_pending > 0`인 동안 `dspqueue_read()`로 응답 대기
- 응답의 `status`, `prof_usecs`, `prof_cycles` 수신 후 `op_pending--`

#### 버퍼 관리 (L219-306)

```
rpcmem_alloc2()  →  [ION/SMMU 공유 메모리 할당]
rpcmem_to_fd()   →  [파일 디스크립터 획득]
fastrpc_mmap()   →  [CPU↔DSP 주소 공간 매핑]
```

- 생성 시 `rpcmem_alloc2`로 **ION 메모리** 할당
- 첫 텐서 초기화 시 `fastrpc_mmap`으로 DSP 도메인에 매핑
- 해제 시 `fastrpc_munmap` → `rpcmem_free`
- **alignment: 128 bytes** (HVX 벡터 크기), **max size: 1GB**

#### Repack 로직 (L340-1362)

HVX에 최적화된 데이터 레이아웃 변환:

| 포맷 | 설명 | 라인 |
|------|------|------|
| **Q4x4x2** | Q4_0 블록 → 4행×2인터리브 패킹 | L340-L631 |
| **Q8x4x2** | Q8_0 블록 → 4행×2인터리브 패킹 | L686-L957 |
| **MXFP4x4x2** | MXFP4 블록 → 4행×2인터리브 패킹 | L1012-L1362 |

`set_tensor` 시 자동 repack, `get_tensor` 시 역변환.

#### 세션 초기화 흐름 (L1535-1680)

```
1. remote_session_control(FASTRPC_RESERVE_NEW_SESSION)  // 새 세션 예약
2. remote_session_control(FASTRPC_GET_URI)              // skel URI 획득
3. remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)// Unsigned PD 활성화
4. htp_iface_open(uri, &handle)                         // FastRPC로 DSP skel 로드  ← 유일한 "전통적 RPC"
5. remote_handle64_control(DSPRPC_CONTROL_LATENCY)      // QoS 모드 활성화
6. dspqueue_create(domain, ..., &queue)                 // 공유메모리 큐 생성 (128KB req / 64KB rsp)
7. dspqueue_export(queue, &queue_id)                    // 큐 ID 익스포트
8. htp_iface_start(handle, dev_id, queue_id, nhvx)     // DSP측 큐 임포트 + 콜백 등록 ← RPC
```

**초기화 때만 FastRPC가 발생**하고, 이후 실제 연산은 dspqueue로 통신.

#### DSP 요청 구성 (L2295-2520)

```
init_binary_req()        → HTP_OP_MUL_MAT / ADD / MUL / SUB / DIV (3 bufs)
init_binary_id_req()     → HTP_OP_MUL_MAT_ID / ADD_ID            (4 bufs)
init_unary_req()         → HTP_OP_RMS_NORM / SCALE / SILU / etc  (2-3 bufs)
init_rope_req()          → HTP_OP_ROPE                            (3-4 bufs)
init_flash_attn_ext_req()→ HTP_OP_FLASH_ATTN_EXT                 (4-6 bufs)
init_cpy_req()           → HTP_OP_CPY                             (2 bufs)
init_set_rows_req()      → HTP_OP_SET_ROWS                        (3 bufs)
init_get_rows_req()      → HTP_OP_GET_ROWS                        (3 bufs)
init_argsort_req()       → HTP_OP_ARGSORT                         (2 bufs)
init_sum_rows_req()      → HTP_OP_SUM_ROWS                        (2 bufs)
init_ssm_conv_req()      → HTP_OP_SSM_CONV                        (3 bufs, src1=CONSTANT)
```

#### Graph Optimize (L2681-2846)

**노드 퓨전** (L2791-L2830):
- `ADD → MUL → RMS_NORM` 같은 연속 Op을 하나의 `node_info`로 묶음
- `ggml_can_fuse()`로 퓨전 가능 여부 확인

**리오더링** (L2730-L2776):
- **동일한 src1을 사용하는 MUL_MAT 노드를 연속 배치**
- VTCM에 양자화된 src1을 한 번만 올려놓고 재사용
- 탐색 범위: 현재 노드에서 +16 개 노드까지

#### 전체 데이터 흐름 다이어그램

```
[llama.cpp 상위 레이어]
        │
        ▼
┌──── supports_op() ──────────────────────────────────────┐
│  Op/타입/shape 조건 검사 → 지원 여부 반환                │
└──────────────────────────────────────────────────────────┘
        │ (지원하는 Op만 Hexagon 그래프에 포함)
        ▼
┌──── graph_optimize() ───────────────────────────────────┐
│  1) 노드 퓨전 (ADD+MUL+RMS_NORM 등)                     │
│  2) 리오더링 (동일 src1 MUL_MAT 연속 배치)               │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──── graph_compute() ────────────────────────────────────┐
│  for each node:                                          │
│    dispatch_op → dspqueue_write(req + buf refs)          │
│                   [비동기, zero-copy]                     │
│  sess->flush() → dspqueue_read(rsp) 대기                 │
└──────────────────────────────────────────────────────────┘
        │ dspqueue (공유메모리 큐)        │
        ▼                                ▼
┌──── HTP (DSP측) htp_packet_callback ────────────────────┐
│  dspqueue_read_noblock → switch(req.op):                 │
│    proc_matmul_req  → vtcm_acquire → 연산 → 응답        │
│    proc_unary_req   → vtcm_acquire → 연산 → 응답        │
│    proc_flash_attn  → vtcm_acquire → 연산 → 응답        │
│    ...                                                   │
│  dspqueue_write(rsp + cache flags)                       │
└──────────────────────────────────────────────────────────┘
```

#### 보조 파일 구조 (`htp/` 디렉토리)

| 파일 | 역할 |
|------|------|
| `htp/main.c` | DSP측 메인: 패킷 콜백, Op 디스패치 |
| `htp/matmul-ops.c` | MUL_MAT 구현 (HVX 타일링) |
| `htp/flash-attn-ops.c` | Flash Attention (VTCM 타일링) |
| `htp/binary-ops.c` | ADD/MUL/SUB/DIV |
| `htp/unary-ops.c` | RMS_NORM, SCALE, SQR, SQRT |
| `htp/act-ops.c` | SILU, GELU, GLU |
| `htp/softmax-ops.c` | Softmax |
| `htp/rope-ops.c` | RoPE |
| `htp/hex-dma.c` | DMA 전송 (DDR↔VTCM) |
| `htp/worker-pool.c` | HVX 워커 스레드 풀 |
| `htp/hvx-*.h` | HVX intrinsic 래퍼 (산술, 복사, exp, sqrt 등) |
| `htp/htp-msg.h` | CPU↔DSP 메시지 프로토콜 정의 |

---

<a id="q3"></a>
## Q3. dspqueue 오버헤드 예상치

### 질문

> `ggml_backend_hexagon_graph_compute()`는 결국 RPC로 HTP에 요청하고 처리 결과를 받는 것인데, 각 op마다 비동기로 동작한다고 했음. 그럼 매 노드마다 dspqueue로 통신을 한다고 했는데 dspqueue overhead는 어느정도 예상할 수 있나?

### 답변

#### 메시지 크기

`htp-msg.h` (L118-L153) 에서 계산되는 실제 크기:

| 항목 | 크기 |
|------|------|
| `htp_tensor` | 40 bytes (data + type + ne[4] + nb[4]) |
| **`htp_general_req`** | **312 bytes** (op + op_params[64B] + flags + 6×htp_tensor) |
| **`htp_general_rsp`** | **64 bytes** (64B 캐시라인 정렬) |
| `dspqueue_buffer` (버퍼 참조 1개) | ~24-32 bytes (fd + ptr + offset + size + flags) |

**노드 1개당 전송량**: ~312B 메시지 + 2~6개 버퍼 참조(~64~192B) = **~400~500 bytes**

#### `dspqueue_write` 1회 오버헤드 추정

dspqueue는 Qualcomm Hexagon SDK의 **공유 메모리 링 버퍼** 기반 메시지 큐. `dspqueue_write`가 수행하는 작업:

```
1. 링 버퍼에 메시지(~400B) memcpy          → ~100-200ns
2. 버퍼 참조 메타데이터 기록                  → ~50-100ns
3. 캐시 유지보수 (cache flush/invalidate)    → ~200-500ns
4. DSP 시그널 (doorbell / interrupt)         → ~500ns-2μs
```

**추정 `dspqueue_write` 레이턴시: ~1~3μs (마이크로초)**

프로파일링 코드 확인 — `dispatch_op` (L2268-L2292)에서 `ggml_time_us()` 전후 차이를 `call_usec`으로 측정:

```cpp
uint64_t t = ggml_time_us();
// ... req 구성, enqueue ...
t = ggml_time_us() - t;  // ← 이것이 call_usec (비동기 모드에서는 순수 enqueue 비용만 포함)
```

#### 전체 그래프에 대한 오버헤드

LLM 추론 1 토큰 기준:

| 항목 | 값 |
|------|-----|
| 일반적인 노드 수 | 100~300개 (모델 크기에 따라) |
| 노드당 enqueue 비용 | ~1-3μs |
| **총 enqueue 오버헤드** | **~100-900μs (0.1~0.9ms)** |
| flush 대기 (마지막 응답 수신) | ~2-5μs (마지막 패킷만) |
| **비교: MUL_MAT 1회 HTP 실행** | **~100-5000μs (Op 크기에 따라)** |

#### 큐 용량 vs 처리 속도

```
Request 큐:  128KB ÷ 312B = ~420 메시지 수용 가능
Response 큐:  64KB ÷  64B = ~1024 메시지 수용 가능
```

일반적인 Transformer 레이어의 노드 수(~20-30개)를 감안하면 **큐가 넘치지 않는다**.

#### 오버헤드가 문제가 되는 경우

| 시나리오 | 오버헤드 비중 | 설명 |
|----------|-------------|------|
| **큰 MUL_MAT** (예: 4096×4096) | **무시할 수준 (<1%)** | Op 실행 시간 자체가 수 ms. 3μs enqueue는 미미 |
| **작은 element-wise Op** (SCALE, SQR) | **5~30%** | Op 실행이 10-20μs일 때 3μs enqueue는 주목할만 |
| **GGML_HEXAGON_OPSYNC=1** (동기 모드) | **심각** | 매 노드마다 write→실행→read 왕복. flush에서 `dspqueue_read` 블로킹 대기 추가 (~5-10μs/회) |

#### 오버헤드 완화 설계 (기존 코드)

1. **비동기 파이프라인** (`opt_opsync=0`): CPU가 연속으로 enqueue, DSP가 콜백에서 병렬 처리. flush는 그래프 끝에서 한 번만.
2. **EARLY_WAKEUP** (L2578): 마지막 Op에만 설정 → DSP가 연산 완료 전에 CPU를 미리 깨움 → flush 대기 시간 단축.
3. **Graph Optimize 리오더링** (L2730): 동일 src1 MUL_MAT를 연속 배치 → `SKIP_QUANTIZE` 플래그로 DSP측 양자화 건너뜀.
4. **Zero-copy 버퍼**: `dspqueue_buffer`에는 fd+offset만 전달. 실제 데이터는 이미 공유 메모리에 있으므로 memcpy 없음.

#### 결론

> 비동기 모드에서 dspqueue 오버헤드는 노드당 ~1-3μs 수준이며, 전체 그래프 기준 0.1~0.9ms. MUL_MAT이나 Flash Attention 같은 무거운 Op이 지배적인 LLM 추론에서는 전체 실행 시간의 **1% 미만**으로 실질적 병목이 아니다. 다만 SCALE, SQR 같은 경량 Op이 많은 경우에는 무시할 수 없는 비중이 될 수 있다.

---

<a id="q4"></a>
## Q4. 비동기 큐 처리에서 그래프 최적화가 필요한 이유

### 질문

> 비동기 deque 처리로 오버헤드가 적다면, 굳이 graph node에 대해서 최적화를 할 필요가 있나?

### 답변

**그래프 최적화의 목적은 dspqueue 오버헤드 감소가 아니다. DSP 내부 실행 비용을 줄이는 것이다.**

#### 리오더링의 실제 목적: VTCM 내 양자화(quantize) 재사용

코드의 핵심 코멘트 (`ggml-hexagon.cpp` L2735):

```
// The main goal here is to stack the MUL_MAT ops with the same src1 input.
// This allows us to reuse dynamically quantized src1 in VTCM.
```

**최적화 없이 (원래 순서):**

```
MUL_MAT(W_q, X)  → [DDR→VTCM DMA] → [F32→Q8 양자화] → [연산]
ADD(...)
MUL_MAT(W_k, X)  → [DDR→VTCM DMA] → [F32→Q8 양자화] → [연산]  ← X 동일한데 재양자화!
RMS_NORM(...)
MUL_MAT(W_v, X)  → [DDR→VTCM DMA] → [F32→Q8 양자화] → [연산]  ← 또 재양자화!
```

**최적화 후 (리오더링):**

```
MUL_MAT(W_q, X)  → [DDR→VTCM DMA] → [F32→Q8 양자화] → [연산]
MUL_MAT(W_k, X)  → [VTCM에 이미 있음, SKIP_QUANTIZE] → [연산]  ← 절약!
MUL_MAT(W_v, X)  → [VTCM에 이미 있음, SKIP_QUANTIZE] → [연산]  ← 절약!
ADD(...)
RMS_NORM(...)
```

#### 양자화 비용의 크기

`matmul-ops.c` L2148-L2195의 `quantize_f32_q8x4x2` 함수:
- 각 행마다: DDR → L2 fetch, HVX copy, F32→Q8 양자화 루프 수행
- src1이 4096차원 × 1행이면 ~16KB의 F32 → ~4.5KB의 Q8x4x2
- **DMA(DDR→VTCM) + HVX 양자화에 수십~수백 μs 소요**

같은 src1을 사용하는 MUL_MAT이 3개라면:
- 리오더링으로 **양자화 2회 절약 = 수십~수백 μs 절약**
- 이것은 dspqueue 1회 오버헤드(~1-3μs)보다 **10~100배 큰 비용**

#### 노드 퓨전의 목적: Op 단위 고정 비용 감소

각 DSP Op에는 고정 오버헤드:

```
Op 1개 실행 시:
  1. vtcm_acquire()     ← VTCM 락 획득
  2. DMA 셋업           ← 스크래치패드 할당
  3. 실제 연산           ← HVX/HTP
  4. DMA 결과 writeback  ← VTCM→DDR
  5. vtcm_release()     ← VTCM 락 해제
  6. send_htp_rsp()     ← 응답 전송
```

퓨전하면 `ADD → MUL → RMS_NORM`을 하나의 Op으로 처리하므로 1, 2, 4, 5, 6 단계가 1번만 실행.

#### 비용 비교 요약

| 비용 요소 | 예상 시간 | 최적화 대상 |
|-----------|----------|------------|
| dspqueue_write (노드당) | ~1-3μs | 이미 충분히 작음 |
| **F32→Q8 양자화** (src1 1회) | **~20-200μs** | **리오더링으로 절약** |
| **DDR→VTCM DMA** (src1 로드) | **~10-50μs** | **리오더링으로 절약** |
| VTCM acquire/release (Op당) | ~1-5μs | 퓨전으로 절약 |
| 응답 전송 (Op당) | ~1-2μs | 퓨전으로 절약 |
| **MUL_MAT 연산 자체** | **~100-5000μs** | 변하지 않음 |

#### 핵심 결론

> **dspqueue 오버헤드가 작다는 것과 그래프 최적화의 필요성은 별개의 문제이다.**
>
> - dspqueue 오버헤드 → **CPU↔DSP 통신 비용** (이미 작음, 최적화 불필요)
> - 그래프 리오더링 → **DSP 내부에서 VTCM 양자화/DMA 재사용** (노드당 수십~수백μs 절약)
> - 그래프 퓨전 → **DSP 내부에서 VTCM 락/DMA 셋업 횟수 감소**
>
> 같은 src1을 쓰는 MUL_MAT 3개가 있으면, 리오더링으로 **양자화 2회 × ~100μs = ~200μs**를 아끼는 것이다. 이는 "Op 3개를 1개로 합쳐서 dspqueue 호출 2회 절약(~6μs)"과는 **차원이 다른 이득**이다.

---

## 용어 정리

| 용어 | 설명 |
|------|------|
| **dspqueue** | Qualcomm Hexagon SDK 제공 공유 메모리 링 버퍼 기반 비동기 메시지 큐 |
| **FastRPC** | Qualcomm의 전통적 원격 프로시저 호출. 커널 경유, 동기식 |
| **VTCM** | Vector Tightly Coupled Memory. HTP 내부 고속 SRAM (~4-8MB) |
| **HVX** | Hexagon Vector eXtensions. 128B 폭 SIMD 벡터 유닛 |
| **ION 메모리** | Android/Qualcomm의 공유 메모리 할당 시스템 (rpcmem 기반) |
| **Repack (x4x2)** | HVX에 최적화된 4행×2인터리브 양자화 블록 레이아웃 |
| **skel** | DSP측에서 로드되는 공유 라이브러리 (예: `libggml-htp-v75.so`) |
| **EARLY_WAKEUP** | 마지막 Op에 설정하여 DSP가 연산 완료 전 CPU를 미리 깨우는 메커니즘 |
