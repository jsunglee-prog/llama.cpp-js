# htp-ops-lib DMA 큐 변환 작업 문서

## 1. 개요

htp-ops-lib의 DMA 전송 방식을 기존 **단일 디스크립터 방식** (`dma_utils.h`)에서 llama.cpp의 **큐 기반 체이닝 방식** (`hex-dma.h/c` 패턴)으로 변환한 작업 내역입니다.

### 배경

| 항목 | 기존 (`dma_utils.h`) | 변환 후 (`dma_queue.h`) |
|------|---------------------|------------------------|
| 디스크립터 | 단일 1D/2D 디스크립터 | 링 버퍼 기반 디스크립터 큐 |
| 전송 시작 | `dmstart` → `dmpoll` → `dmwait` | `dmstart` + `dmlink` 체이닝 |
| 파이프라이닝 | 불가 (idle 상태에서만 submit) | 가능 (push 여러 개 → pop으로 순차 완료 대기) |
| 동기화 | `dma_wait_for_idle()` 전체 대기 | `dstate` 플래그로 개별 디스크립터 완료 확인 |
| fallback | HVX memcpy (`use_dma = false`) | 없음 (DMA 전용) |
| 참조 구현 | htp-ops-lib 자체 | llama.cpp `ggml/src/ggml-hexagon/htp/hex-dma.h/c` |

---

## 2. 파일 변경 목록

### 2.1 새로 생성된 파일

#### `htp-ops-lib/include/dsp/dma_queue.h` (206줄)

큐 기반 DMA API 헤더. llama.cpp의 `hex-dma.h`를 htp-ops-lib 환경에 맞게 적용.

**주요 구성:**

```
┌─────────────────────────────────────────────────────┐
│ dma_queue_desc_t (Type-1, 2D 디스크립터)              │
│  - next, src, dst, roiwidth, roiheight, stride 등   │
│  - dstate: 완료 여부 (HW가 1로 설정)                   │
│  - 64바이트 정렬                                      │
├─────────────────────────────────────────────────────┤
│ dma_queue_t (큐 구조체)                               │
│  - desc[]   : 디스크립터 링 버퍼                       │
│  - dptr[]   : dst/src 포인터 북키핑                    │
│  - tail     : 마지막 링크된 디스크립터 (dmlink 타겟)    │
│  - push_idx / pop_idx / idx_mask (power-of-2 링)     │
├─────────────────────────────────────────────────────┤
│ Inline 함수                                          │
│  - dma_queue_push()       : 2D 전송 enqueue + dmlink │
│  - dma_queue_pop()        : dstate spin-wait + dequeue│
│  - dma_queue_push_ddr_to_vtcm_1d() : 1D 편의 함수    │
│  - dma_queue_push_ddr_to_vtcm_2d() : 2D 편의 함수    │
│  - dma_queue_empty() / dma_queue_depth()             │
├─────────────────────────────────────────────────────┤
│ ASM 래퍼                                             │
│  - dma_q_dmstart() : release + dmstart               │
│  - dma_q_dmlink()  : release + dmlink                │
│  - dma_q_dmpoll()  : dmpoll                          │
│  - dma_q_dmwait()  : dmwait                          │
└─────────────────────────────────────────────────────┘
```

**핵심 흐름 (`push` → `pop`):**

```
push:                              pop:
  desc 필드 설정                      spin { dmpoll(); check dstate }
  dstate = INCOMPLETE                dptr 반환
  dmlink(tail, desc)                 pop_idx++
  tail = desc
  push_idx++
```

#### `htp-ops-lib/src/dsp/dma_queue.c` (68줄)

큐 생명주기 관리 구현.

| 함수 | 설명 |
|------|------|
| `dma_queue_create(capacity)` | capacity를 2의 거듭제곱으로 올림 → `memalign(64)`로 디스크립터 배열 할당 → tail을 마지막 슬롯(센티넬)으로 초기화 |
| `dma_queue_delete(q)` | desc, dptr, q 순서로 free |
| `dma_queue_flush(q)` | `dma_queue_pop()` 반복하여 대기중인 전송 모두 완료 |

---

### 2.2 수정된 파일

#### `htp-ops-lib/src/dsp/ops/mat_mul.c`

**변경 요약:** `#include "dsp/dma_utils.h"` → `"dsp/dma_queue.h"`, 5개 DMA 패턴 변환

##### 패턴 1: `transfer_permuted_weight_chunk_fp16()` (L171~182)

FP16 가중치 청크를 DDR → VTCM으로 1D 동기 전송.

```c
// Before (dma_utils.h):
dma_issue_load_from_ddr(vtcm_dst, src, size);
dma_wait_for_idle();

// After (dma_queue.h):
dma_queue_push_ddr_to_vtcm_1d(g_dma_queue, dma_make_ptr(vtcm_dst, src), size);
dma_queue_pop(g_dma_queue);
```

##### 패턴 2: 더블 버퍼 파이프라인 (L935~970)

양자화 가중치를 청크 단위로 프리페치하며 처리하는 루프.

```c
// Before:
dma_issue_load_from_ddr(buf_curr, permuted_weight, first_size);
// loop:
  dma_wait_for_idle();
  dma_issue_load_from_ddr(buf_next, next_chunk, next_size);  // prefetch
  // ...compute on buf_curr...
  swap(buf_curr, buf_next);

// After:
dma_queue_push_ddr_to_vtcm_1d(g_dma_queue, dma_make_ptr(buf_curr, permuted_weight), first_size);
// loop:
  dma_queue_pop(g_dma_queue);           // wait current
  dma_queue_push_ddr_to_vtcm_1d(g_dma_queue, dma_make_ptr(buf_next, next), next_size); // prefetch
  // ...compute on buf_curr...
  swap(buf_curr, buf_next);
```

##### 패턴 3: 4단계 파이프라인 A→B→C→D (L1015~1070)

out-stationary 매트릭스 곱셈의 프롤로그/메인 루프.

```
A = DMA load (양자화 가중치)
B = dequantize (HVX)
C = HMX core dot product
D = output store

prologue:
  push(A0) → pop(A0) → B0 → push(A1)
main loop:
  pop(A_i) → B_i → push(A_{i+1}) → C_{i-1} → D_{i-2}
```

변환 방식은 패턴 2와 동일하게 `push` + `pop` 로 대체.

##### 패턴 4: 2D DMA `dma_load_2d_sync` (g_dma_queue_fetch 사용)

fetch 워커 스레드에서 사용하는 별도 큐.

```c
// Before:
dma_desc_2d_t desc;
// ... manual desc setup row by row ...
dma_submit_one(&desc);
dma_wait_for_idle();

// After:
dma_queue_push_ddr_to_vtcm_2d(g_dma_queue_fetch, dma_make_ptr(dst, src),
                               width, height, src_stride, dst_stride);
dma_queue_pop(g_dma_queue_fetch);
```

##### 패턴 5: out-stationary 수동 2D 디스크립터

```c
// Before:
dma_desc_2d_t desc __attribute__((aligned(64)));
desc.type = DMA_DESC_TYPE_2D;
desc.roi_width = ...;
// ... 10+ lines of field setup ...
dmstart(&desc);
dmwait();

// After:
dma_queue_push_ddr_to_vtcm_2d(g_dma_queue, dptr, width, height, src_stride, dst_stride);
dma_queue_pop(g_dma_queue);
```

#### `htp-ops-lib/src/dsp/ops/flash_attn.c`

**변경 요약:** Q 타일 로딩의 수동 2D DMA → `dma_queue_push_ddr_to_vtcm_2d`

##### Q 타일 로딩 (L917~924)

Flash Attention에서 Query 타일을 DDR → VTCM으로 2D 전송.

```c
// Before:
dma_desc_2d_t desc __attribute__((aligned(64)));
desc.src = q_ld_base;
desc.dst = q_tile;
desc.roi_width = qo_ldst_blk_sz * sizeof(float);
desc.roi_height = n_rows;
desc.src_stride = qo_ldst_stride * sizeof(float);
desc.dst_stride = qo_ldst_blk_sz * sizeof(float);
// ... more field setup ...
dmstart(&desc);
dmwait();

// After:
extern dma_queue_t *g_dma_queue;
dma_queue_push_ddr_to_vtcm_2d(g_dma_queue,
                              dma_make_ptr(q_tile, q_ld_base),
                              qo_ldst_blk_sz * sizeof(float),   // width
                              n_rows,                            // height
                              qo_ldst_stride * sizeof(float),   // src_stride
                              qo_ldst_blk_sz * sizeof(float));  // dst_stride
dma_queue_pop(g_dma_queue);
```

##### O 타일 저장 (L1296~1303, 주석 처리)

VTCM → DDR 역방향 DMA는 조사 필요 사항으로 주석으로만 남겨둠.

```c
// TODO(hzx): investigate why DMA is not working here
// extern dma_queue_t *g_dma_queue;
// dma_queue_push_ddr_to_vtcm_2d(g_dma_queue,
//                               dma_make_ptr(o_st_base, o_tile), ...);
// dma_queue_pop(g_dma_queue);
```

#### `htp-ops-lib/src/dsp/commu.c`

**변경 요약:** 전역 DMA 큐 생성/정리 코드 추가

```c
// 추가된 include
#include "dsp/dma_queue.h"

// 전역 변수 선언 (L28-29)
dma_queue_t *g_dma_queue       = NULL;  // main thread DMA
dma_queue_t *g_dma_queue_fetch = NULL;  // fetch worker thread DMA
```

| 함수 | 추가 내용 |
|------|----------|
| `htp_ops_init_backend()` (L264~267) | `g_dma_queue = dma_queue_create(64)` / `g_dma_queue_fetch = dma_queue_create(64)` + 에러 체크 |
| `htp_ops_close()` (L239~248) | `dma_queue_flush()` → `dma_queue_delete()` → `NULL` 설정 (각 큐에 대해) |

**두 개의 큐를 사용하는 이유:**
- `g_dma_queue`: 메인 HW 스레드에서 사용 (mat_mul, flash_attn)
- `g_dma_queue_fetch`: fetch 워커 스레드에서 사용 (별도 HW 스레드 = 별도 DMA 엔진)
- Hexagon CDSP에서 각 HW 스레드는 독립적인 DMA 엔진을 가짐

#### `htp-ops-lib/CMakeLists.txt`

**변경 요약:** 빌드 소스에 `dma_queue.c` 추가

```cmake
add_library(htp_ops_skel SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_skel.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/commu.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/dma_queue.c    # ← 추가
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/power.c
    ...
)
```

---

## 3. 기존 파일 (변경 없음)

### `htp-ops-lib/include/dsp/dma_utils.h` (119줄)

기존 단일 디스크립터 DMA API. 더 이상 어떤 소스 파일에서도 include하지 않으나, 참조용으로 유지.

**제공하던 API:**

| 함수/매크로 | 설명 |
|------------|------|
| `dma_submit_one(desc)` | idle 확인 후 `dmstart` (단일 전송) |
| `dma_wait_for_idle()` | `dmwait` + 상태 확인 |
| `dmstart(next)` | `release` + `Q6_dmstart_A` |
| `dmlink(cur, next)` | `release` + `Q6_dmlink_AA` |
| `dmpoll()` / `dmwait()` | HW 상태 조회 |
| `dma_desc_1d_t` | 1D descriptor 구조체 |
| `dma_desc_2d_t` | 2D descriptor 구조체 |

> **향후 작업:** 더 이상 사용되지 않으므로 삭제 가능.

---

## 4. 아키텍처 비교

### 기존 방식 (dma_utils.h)

```
CPU Thread          DSP DMA Engine
    │                    │
    ├─ dma_submit_one ──►│── transfer ──►│ complete
    │                    │               │
    ├─ dma_wait_for_idle ◄───────────────┘
    │  (전체 DMA 엔진이 idle될 때까지 대기)
    │
    ├─ dma_submit_one ──►│── transfer ──►│ complete
    │                    │               │
    └─ dma_wait_for_idle ◄───────────────┘
```

- **문제점:** idle 확인 후에만 submit 가능 → 파이프라이닝 불가

### 변환 후 (dma_queue.h)

```
CPU Thread          DSP DMA Engine (dmlink chain)
    │                    │
    ├─ push(desc0) ─────►│── transfer0 ──►│ dstate=1
    ├─ push(desc1) ─────►│── transfer1 ──►│ dstate=1
    ├─ push(desc2) ─────►│── transfer2 ──►│
    │                    │               │
    ├─ pop() ◄───────────┤ (desc0 완료)   │
    │  [compute on 0]    │               │
    ├─ pop() ◄───────────┤ (desc1 완료)   │
    │  [compute on 1]    │               │
    └─ pop() ◄──────────────────────────┘ (desc2 완료)
```

- **장점:** DMA 전송과 HVX/HMX 연산이 오버랩 가능
- `dmlink`로 체이닝하므로 DMA 엔진이 idle로 돌아가지 않고 연속 처리

---

## 5. 변경 통계

| 항목 | 값 |
|------|-----|
| 변경된 파일 수 | 6 (신규 2 + 수정 4) |
| 추가된 코드 | +342줄 |
| 삭제된 코드 | -113줄 |
| 변환된 DMA 패턴 수 | mat_mul: 5개, flash_attn: 1개 (+1 주석) |
| 참조 구현 | `ggml/src/ggml-hexagon/htp/hex-dma.h` (llama.cpp) |

---

## 6. 참고 사항

### UDMA 하드웨어 특성
- Hexagon CDSP의 UDMA는 **하드웨어 명령어** (`dmstart`, `dmlink`, `dmpoll`, `dmwait`)
- 별도 초기화 불필요 — 명령어 실행 즉시 동작
- 각 HW 스레드마다 독립적인 DMA 엔진 보유

### 알려진 제한사항
- O 타일 VTCM → DDR 역방향 DMA가 동작하지 않는 문제 (flash_attn.c L1296, 원인 조사 필요)
- DMA 실패 시 fallback 없음 (`dma_queue_pop`이 무한 spin) — 필요 시 timeout 로직 추가 고려
- `dma_utils.h`는 참조용으로 남아있으나, 빌드에 포함되지 않음
