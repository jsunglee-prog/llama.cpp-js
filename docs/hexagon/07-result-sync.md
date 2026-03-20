# 7. 연산 결과 회수 및 CPU 동기화 분석

> 분석 대상 코드: [ggml-hexagon.cpp:140-215](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp) (enqueue/flush), [htp/main.c:1044-1200](../../ggml/src/ggml-hexagon/htp/main.c) (응답 전송), [htp/htp-msg.h](../../ggml/src/ggml-hexagon/htp/htp-msg.h) (메시지 포맷)

---

## 7.1 통신 채널: dspqueue

CPU↔DSP 간의 모든 통신은 **dspqueue** (DSP Queue) 를 통해 이루어집니다. 이것은 Qualcomm의 FastRPC 프레임워크 위에 구축된 비동기 메시지 큐입니다.

```
┌───────────┐                              ┌──────────────┐
│   CPU     │    요청 큐 (128KB)           │   DSP (CDSP) │
│           │  ──────────────────────────→  │              │
│  enqueue()│    htp_general_req (64B+)     │ packet_      │
│           │                               │ callback()   │
│  flush()  │    응답 큐 (64KB)            │              │
│   ←read   │  ←─────────────────────────  │ send_htp_    │
│           │    htp_general_rsp (64B)      │ rsp()        │
└───────────┘                              └──────────────┘
```

### 큐 사양

| 속성 | 요청 큐 | 응답 큐 |
|------|---------|---------|
| 크기 | 128KB | 64KB |
| 메시지 크기 | 가변 (req + 버퍼 참조) | 64B 고정 |
| 방향 | CPU → DSP | DSP → CPU |
| 모드 | 비블로킹/블로킹 선택 | 블로킹 read |

---

## 7.2 요청 전송: `enqueue()`

위치: [ggml-hexagon.cpp:141](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)

```cpp
void ggml_hexagon_session::enqueue(struct htp_general_req &req, 
                                    struct dspqueue_buffer *bufs, 
                                    uint32_t n_bufs, bool sync) {
    // 1. 요청을 큐에 기록
    int err = dspqueue_write(queue, 
                             0,              // flags
                             DSPQUEUE_WRITE,  // writes 
                             sizeof(req),     // 요청 크기
                             &req,            // 요청 데이터
                             n_bufs,          // 버퍼 수
                             bufs);           // 버퍼 배열 (fd 기반 참조)
    
    // 2. 대기 Op 카운터 증가
    op_pending.fetch_add(1, std::memory_order_relaxed);
    
    // 3. 동기 모드면 즉시 flush
    if (sync || opt_opsync) {
        flush();
    }
}
```

### 요청 메시지 구조 (`htp_general_req`)

위치: [htp/htp-msg.h](../../ggml/src/ggml-hexagon/htp/htp-msg.h)

```c
struct htp_general_req {
    uint32_t op;             // HTP_OP_MUL_MAT 등 (enum htp_op)
    uint8_t  op_params[64];  // Op별 파라미터 (64바이트)
    uint32_t flags;          // SKIP_QUANTIZE, SKIP_COMPUTE, EARLY_WAKEUP
    
    struct htp_tensor src0;  // 가중치 텐서
    struct htp_tensor src1;  // 입력 텐서
    struct htp_tensor src2;  // (옵션) 추가 입력
    struct htp_tensor src3;  // (옵션) 마스크 등
    struct htp_tensor src4;  // (옵션) 
    struct htp_tensor dst;   // 출력 텐서
};

struct htp_tensor {
    uint32_t data;           // 버퍼 내 오프셋
    uint32_t type;           // 데이터 타입 (HTP_TYPE_F32 등)
    uint32_t ne[4];          // 차원 크기
    uint32_t nb[4];          // 스트라이드
};
```

---

## 7.3 DSP에서 연산 수행 후 응답 전송

위치: [htp/main.c](../../ggml/src/ggml-hexagon/htp/main.c)

### DSP 메인 루프

```c
static int htp_packet_callback(dspqueue_t queue, ...) {
    while (1) {
        // 비블로킹 읽기 시도
        err = dspqueue_read_noblock(queue, 0, sizeof(req), &req, ...);
        
        if (err == AEE_SUCCESS) {
            // 패킷 수신 성공 → Op 디스패치
            switch (req.op) {
                case HTP_OP_MUL_MAT:
                    proc_binary_req(ctx, &req, bufs);  // 행렬곱 처리
                    break;
                case HTP_OP_SOFTMAX:
                    proc_unary_req(ctx, &req, bufs);   // Softmax 처리
                    break;
                // ... 기타 Op들
            }
            
            continue;  // 다음 패킷 즉시 시도
        }
        
        // 큐가 비어있으면 블로킹 대기
        err = dspqueue_read(queue, 0, sizeof(req), &req, ...);
    }
}
```

### Op 처리 과정

```c
static void proc_binary_req(struct htp_context * ctx, 
                             struct htp_general_req * req,
                             struct dspqueue_buffer * bufs) {
    // 1. Op 컨텍스트 구성
    struct htp_ops_context octx = {0};
    octx.ctx = ctx;
    octx.op = req->op;
    memcpy(octx.op_params, req->op_params, sizeof(req->op_params));
    octx.flags = req->flags;
    octx.n_threads = ctx->n_threads;
    
    // 2. 텐서 포인터 매핑 (버퍼 오프셋 → 실제 주소)
    octx.src0 = req->src0;
    octx.src0.data = bufs[0].ptr + req->src0.data;  // 오프셋 → 절대 주소
    octx.src1 = req->src1;
    octx.src1.data = bufs[1].ptr + req->src1.data;
    octx.dst = req->dst;
    octx.dst.data = bufs[dst_idx].ptr + req->dst.data;
    
    // 3. VTCM 획득
    vtcm_acquire(ctx);
    
    // 4. 연산 실행 (워커 풀에서 병렬 실행)
    uint64_t t1 = HAP_perf_get_qtimer_count();
    int status = op_matmul(&octx);  // 또는 다른 Op 함수
    uint64_t t2 = HAP_perf_get_qtimer_count();
    
    // 5. VTCM 해제
    vtcm_release(ctx);
    
    // 6. 응답 전송
    struct htp_general_rsp rsp = {0};
    rsp.op = req->op;
    rsp.status = status;
    rsp.usecs = HAP_perf_qtimer_count_to_us(t2 - t1);
    rsp.cycles = t2 - t1;
    rsp.packets = 1;
    
    send_htp_rsp(ctx, &rsp);
}
```

### 응답 전송

```c
static void send_htp_rsp(struct htp_context * ctx, struct htp_general_rsp * rsp) {
    // 64바이트 응답을 응답 큐에 기록
    int err = dspqueue_write(ctx->queue,
                             0,              // flags
                             DSPQUEUE_WRITE,
                             sizeof(*rsp),   // 64바이트
                             rsp,
                             0, NULL);       // 버퍼 없음
}
```

### 응답 메시지 구조 (`htp_general_rsp`)

```c
struct htp_general_rsp {
    uint32_t op;       // 어떤 Op에 대한 응답인지
    int32_t  status;   // HTP_STATUS_OK, HTP_STATUS_VTCM_TOO_SMALL 등
    uint32_t usecs;    // 실행 시간 (마이크로초)
    uint32_t cycles;   // 실행 사이클 수
    uint32_t packets;  // 패킷 수
    uint8_t  pad[];    // 64바이트까지 패딩
};
```

---

## 7.4 CPU에서 결과 회수: `flush()`

위치: [ggml-hexagon.cpp:164](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)

```cpp
void ggml_hexagon_session::flush() {
    while (op_pending.load(std::memory_order_relaxed) > 0) {
        struct htp_general_rsp rsp;
        
        // 블로킹 읽기: DSP 응답 대기
        int err = dspqueue_read(queue,
                                0,              // flags
                                sizeof(rsp),    // 응답 크기
                                &rsp,           // 응답 버퍼
                                0, NULL);       // 버퍼 없음
        
        if (err != 0) {
            GGML_ABORT("dspqueue_read failed");
        }
        
        // 에러 체크
        if (rsp.status != HTP_STATUS_OK) {
            GGML_ABORT("DSP op failed: status=%d", rsp.status);
        }
        
        // 프로파일링 데이터 수집
        if (opt_profile) {
            prof_usecs  += rsp.usecs;
            prof_cycles += rsp.cycles;
            prof_pkts   += rsp.packets;
        }
        
        // 대기 카운터 감소
        op_pending.fetch_sub(1, std::memory_order_relaxed);
    }
}
```

### 핵심 동기화 포인트

```
graph_compute() 의 흐름:
  │
  ├── Op #0 enqueue() → op_pending = 1 (비동기)
  ├── Op #1 enqueue() → op_pending = 2 (비동기)
  ├── Op #2 enqueue() → op_pending = 3 (비동기)
  │   ...
  ├── Op #N enqueue() + EARLY_WAKEUP 플래그
  │     → DSP가 마지막 Op 결과를 우선 전송
  │
  └── flush()  ← 블로킹! 
        ├── read rsp #0 → op_pending = N-1
        ├── read rsp #1 → op_pending = N-2
        │   ...
        └── read rsp #N → op_pending = 0 → 반환
        
  모든 결과가 DRAM에 확정됨 ✓
```

---

## 7.5 결과 데이터의 물리적 경로

### VTCM → DRAM 기록

DSP 연산 결과는 VTCM에서 직접 DRAM으로 기록됩니다:

```
MatMul의 경우:
  matmul_2d() 내부:
    // 2x2 벡터 내적 결과 → dst 포인터에 직접 기록
    mmctx->vec_dot_2x2(ne00, &dst_data[...], &dst_data[...], ...);
    // dst_data는 DRAM의 dst 텐서를 가리킴
    // HVX store 명령어가 L1/L2 캐시를 통해 DRAM에 기록
    
Flash Attention의 경우:
  // VKQ32 누적기(VTCM)의 최종 결과를 DRAM dst에 기록
  hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
```

### DRAM → CPU 접근

FastRPC mmap으로 인해 CPU와 DSP가 **같은 물리 메모리**를 공유합니다:

```
DSP 연산 완료
  → VTCM → DRAM 기록 (HVX store / L2 캐시 writeback)
  → dspqueue_write(rsp)  ← 응답 전송
  
CPU flush()에서 응답 수신
  → DRAM의 dst 텐서 데이터가 이미 확정됨
  → CPU가 동일 물리 주소를 자신의 가상 주소로 바로 접근
  → 별도의 DMA나 memcpy 없이 즉시 사용 가능
```

### FastRPC 캐시 일관성 (Cache Coherency)

- **ION/DMA-BUF 메모리**: CPU와 DSP 간 하드웨어 캐시 일관성 보장
- `rpcmem_alloc2()`로 할당된 메모리는 **캐시 일관(Coherent)** 영역
- DSP의 HVX store가 DRAM에 기록되면, CPU는 자동으로 최신 데이터를 읽음
- dspqueue의 응답 전송 자체가 **메모리 배리어** 역할을 수행

---

## 7.6 로짓(Logits) 취합 및 다음 토큰 생성

```
graph_compute() 완료 (flush() 반환)
  │
  ├── dst 텐서(로짓)가 DRAM의 Hexagon 버퍼에 확정
  │
  └── 스케줄러 반환 → llama_decode() 반환
        │
        └── 로짓 텐서를 sampling 모듈에 전달
              │
              ├── is_host=true (regular buffer)일 경우:
              │     CPU가 Hexagon 버퍼를 직접 읽음 (제로 카피)
              │
              ├── is_host=false (repack buffer)일 경우:
              │     스케줄러가 자동으로 CPU 버퍼에 복사 후 전달
              │
              └── 토큰 샘플링 (temperature, top-k, top-p 등)
                    → 다음 토큰 ID 결정
                    → 다음 추론 사이클 시작
```

---

## 7.7 EARLY_WAKEUP 최적화

```cpp
// graph_compute() 내에서
if (i == last) {
    flags |= HTP_OPFLAGS_EARLY_WAKEUP;
}
```

마지막 Op에 `EARLY_WAKEUP` 플래그를 설정하면:
1. DSP가 해당 Op의 응답을 일반 Op보다 **더 빨리** 전송
2. CPU의 `flush()` 가 더 빨리 깨어남
3. CPU↔DSP 간의 레이턴시 오버헤드 감소

이는 dspqueue의 배칭 동작을 무시하고 즉각적인 응답을 강제하는 최적화입니다.

---

## 7.8 에러 상태 처리

| 상태 코드 | 의미 | 대응 |
|----------|------|------|
| `HTP_STATUS_OK` | 성공 | 정상 진행 |
| `HTP_STATUS_INTERNAL_ERR` | DSP 내부 오류 | ABORT |
| `HTP_STATUS_NO_SUPPORT` | 미지원 Op/타입 | ABORT (supports_op에서 이미 필터링했어야 함) |
| `HTP_STATUS_INVAL_PARAMS` | 잘못된 파라미터 | ABORT |
| `HTP_STATUS_VTCM_TOO_SMALL` | VTCM 부족 | ABORT (텐서가 VTCM에 맞지 않음) |
