# 3. llama-cli → Hexagon DSP 오프로딩 과정 (Dispatch Pipeline)

> 분석 대상 코드: [ggml-hexagon.cpp](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp), [htp-drv.cpp](../../ggml/src/ggml-hexagon/htp-drv.cpp), [htp/main.c](../../ggml/src/ggml-hexagon/htp/main.c)

---

## 3.1 초기화 체인: 등록 → 세션 생성 → DSP 연결

### 단계 1: 백엔드 레지스트리 등록

```
ggml_backend_hexagon_reg()                          // ggml-hexagon.cpp:3274
  └── htpdrv_init()                                 // htp-drv.cpp
  │     ├── dlopen("libcdsprpc.so")                 // FastRPC 드라이버 로드
  │     └── 함수 포인터 바인딩:
  │           rpcmem_alloc2, rpcmem_free, rpcmem_to_fd
  │           fastrpc_mmap2, fastrpc_munmap
  │           dspqueue_create, dspqueue_close, dspqueue_export
  │           dspqueue_write, dspqueue_read
  │           remote_handle64_open, remote_handle64_invoke
  │           remote_session_control
  │
  └── ggml_hexagon_init(&reg)                       // ggml-hexagon.cpp:3244
        ├── 환경변수 파싱 (GGML_HEXAGON_*)
        ├── get_hex_arch_ver() → DSP 아키텍처 버전 확인 (v73, v75 등)
        └── new ggml_hexagon_registry(reg)          // ggml-hexagon.cpp:3173
              └── for i in 0..opt_ndev:
                    new ggml_hexagon_session(i, &devices[i])
```

### 단계 2: 세션 할당 (CPU↔DSP 연결)

```
ggml_hexagon_session::allocate(dev_id)              // ggml-hexagon.cpp:1535
  │
  ├── 1. get_domain(3)                              // CDSP 도메인 획득
  │
  ├── 2. remote_session_control(RESERVE_NEW_SESSION) // 새 FastRPC 세션 예약
  │     → session_id, effective_domain_id 반환
  │
  ├── 3. remote_session_control(UNSIGNED_MODULE)     // Unsigned PD 활성화
  │
  ├── 4. htp_iface_open(uri, &handle)               // DSP 핸들 열기
  │     → uri = "file:///libggml-htp-v{arch}.so?htp_iface_skel_handle_invoke"
  │     → DSP에서 libggml-htp-v{arch}.so 로딩됨
  │
  ├── 5. remote_handle64_control(DSPRPC_CONTROL_LATENCY) // FastRPC QoS 모드 활성화
  │
  ├── 6. dspqueue_create(domain_id, ...)            // dspqueue 생성
  │     → 요청 큐: 128KB
  │     → 응답 큐: 64KB
  │
  ├── 7. dspqueue_export(queue, &queue_id)          // 큐 ID 내보내기
  │
  └── 8. htp_iface_start(handle, dev_id, queue_id, nhvx)  // DSP 서비스 시작
        → DSP측에서 htp_iface_start() 호출됨
```

### 단계 3: DSP측 초기화

```
htp_iface_start()                                    // htp/main.c
  ├── dspqueue_import(queue_id, packet_callback)     // 큐 임포트 + 콜백 등록
  ├── vtcm_alloc(ctx)                                // VTCM 할당 (기본 8MB)
  │     ├── HAP_compute_res_attr_init()
  │     ├── HAP_compute_res_attr_set_vtcm_param(size, 1 /*cached*/)
  │     ├── HAP_compute_res_attr_set_hmx(1)          // HMX 활성화
  │     └── HAP_compute_res_acquire()
  ├── DMA 큐 초기화 (스레드당 64엔트리)
  └── worker_pool_init(n_hvx)                        // 워커 스레드 풀 생성
```

---

## 3.2 추론 실행: 그래프 → DSP 오프로딩

### CPU측: `graph_compute()`

위치: [ggml-hexagon.cpp:2569](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)

```cpp
static enum ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    // 각 노드를 순회하면서 DSP에 Op을 전송
    for (int i = first; i <= last; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        uint32_t flags = 0;

        // src1 재사용 최적화: 이전 Op과 같은 src1이면 양자화 스킵
        if (op_reuse_src1(node, prev_op)) {
            flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
        }

        // 마지막 Op에 조기 알림 플래그
        if (i == last) {
            flags |= HTP_OPFLAGS_EARLY_WAKEUP;
        }

        // Op 종류에 따라 dispatch
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_hexagon_dispatch_op<init_binary_req<true>>(sess, node, flags);
                break;
            // ... 다른 Op들 ...
        }
    }

    // 모든 Op 완료 대기
    sess->flush();
    return GGML_STATUS_SUCCESS;
}
```

### Dispatch 과정 상세

`ggml_hexagon_dispatch_op<>` 는 템플릿 함수로, `init_req` 함수를 통해 요청 패킷을 구성합니다:

```
ggml_hexagon_dispatch_op<init_binary_req<true>>(sess, node, flags)
  │
  ├── 1. htp_general_req 구조체 초기화
  │     ├── req.op = HTP_OP_MUL_MAT
  │     ├── req.flags = flags
  │     ├── req.op_params[64] = 노드별 파라미터 (op_params 복사)
  │     ├── req.src0 = {data=오프셋, type, ne[4], nb[4]}  // 가중치 텐서
  │     ├── req.src1 = {data=오프셋, type, ne[4], nb[4]}  // 입력 텐서
  │     └── req.dst  = {data=오프셋, type, ne[4], nb[4]}  // 출력 텐서
  │
  ├── 2. dspqueue_buffer[] 배열 구성
  │     └── 각 텐서의 실제 메모리 버퍼 참조 (fd 기반)
  │
  └── 3. sess->enqueue(req, bufs, n_bufs, sync)
        ├── dspqueue_write(queue, req, bufs, n_bufs)  // 큐에 기록
        └── op_pending.fetch_add(1)                    // 대기 Op 수 증가
```

### DSP측: 패킷 수신 및 처리

```
htp_packet_callback()                                // htp/main.c:1044
  └── while(1):
        dspqueue_read_noblock(queue, &req, bufs)     // 비블로킹 읽기
        │
        ├── 성공 시:
        │     ├── req.op에 따라 proc_*_req() 호출
        │     │     ├── htp_ops_context 구성 (텐서 포인터 + 버퍼 매핑)
        │     │     ├── vtcm_acquire(ctx)             // VTCM 확보
        │     │     ├── op_function(octx)             // 실제 연산 실행
        │     │     ├── vtcm_release(ctx)             // VTCM 해제
        │     │     └── send_htp_rsp(rsp)             // 응답 전송
        │     │
        │     └── 다음 패킷 읽기 시도
        │
        └── 큐 비어있으면:
              dspqueue_read(queue, ...)               // 블로킹 대기
```

---

## 3.3 `ggml_backend_t` 인터페이스를 통한 CPU→Hexagon 전환 (Dispatch 과정)

### supports_op()에 의한 라우팅

스케줄러가 그래프를 분할할 때, 각 노드에 대해 `ggml_backend_hexagon_device_supports_op()`을 호출합니다:

```cpp
// ggml-hexagon.cpp:2984
static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);

    // 1단계: 모든 src/dst 버퍼가 이 세션에 속하는지 확인
    if (!ggml_hexagon_supported_buffers(sess, op)) {
        return false;
    }

    // 2단계: Op별 세부 지원 여부 확인
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return ggml_hexagon_supported_mul_mat(sess, op);
        // ...
    }
}
```

### 버퍼 할당을 통한 자동 매핑

모델 가중치가 Hexagon 버퍼에 할당되면, 스케줄러는 해당 텐서를 사용하는 Op을 자동으로 Hexagon에 할당합니다. 이것이 CPU→Hexagon 전환의 핵심 메커니즘입니다:

```
모델 로딩 시:
  가중치 텐서 → ggml_backend_hexagon_buffer_alloc() → rpcmem_alloc2() + fastrpc_mmap2()
  → 텐서의 buffer가 Hexagon 타입

스케줄러 패스 1:
  node의 src[0]가 Hexagon 버퍼 → node를 Hexagon 백엔드로 할당

패스 2~4:
  인접 Op들도 Hexagon이 지원하면 → Hexagon으로 확장
  미지원 Op → CPU 폴백
```

---

## 3.4 환경변수를 통한 동작 제어

| 환경변수 | 기본값 | 설명 |
|---------|--------|------|
| `GGML_HEXAGON_NDEV` | 1 | 동시 DSP 세션 수 (최대 16) |
| `GGML_HEXAGON_NHVX` | (auto) | HVX 스레드 수 |
| `GGML_HEXAGON_HOSTBUF` | 1 | 호스트 버퍼 사용 (0=DSP 전용 버퍼) |
| `GGML_HEXAGON_OPMASK` | (all) | Op 마스크 (특정 Op만 활성화) |
| `GGML_HEXAGON_OPSYNC` | 0 | 동기 모드 (디버깅용) |
| `GGML_HEXAGON_PROFILE` | 0 | 프로파일링 활성화 |
| `GGML_HEXAGON_ARCH` | (auto) | 강제 아키텍처 지정 (v73, v75 등) |
| `GGML_HEXAGON_VERBOSE` | 0 | 상세 로그 출력 |
| `GGML_HEXAGON_ETM` | 0 | ETM 하드웨어 트레이싱 활성화 |
