# 1. ggml 라이브러리 내 하드웨어 가속(Backend) 추상화 방식

> 분석 대상 코드: [ggml/src/ggml-backend.cpp](../../ggml/src/ggml-backend.cpp), [ggml/src/ggml-backend-reg.cpp](../../ggml/src/ggml-backend-reg.cpp), [ggml/include/ggml-backend.h](../../ggml/include/ggml-backend.h)

---

## 1.1 백엔드 추상화 계층 구조

ggml은 **4단계 인터페이스 계층**을 사용하여 하드웨어를 추상화합니다:

```
ggml_backend_reg_t (Registry)          ← 백엔드 전체 등록 (드라이버 로딩)
  └── ggml_backend_dev_t (Device)      ← 개별 디바이스 (GPU #0, DSP #1 등)
       └── ggml_backend_t (Backend)    ← 실행 인스턴스 (큐, 세션)
            └── ggml_backend_buffer_t  ← 메모리 버퍼 (디바이스별 할당)
```

### 각 계층의 역할

| 계층 | 인터페이스 | 주요 역할 |
|------|-----------|---------|
| **Registry** (`ggml_backend_reg_i`) | `get_device_count()`, `get_device()`, `get_proc_address()` | 백엔드 최초 등록, 디바이스 열거 |
| **Device** (`ggml_backend_device_i`) | `supports_op()`, `supports_buft()`, `init_backend()`, `get_buffer_type()` | Op 지원 여부 판단, 버퍼 타입 제공 |
| **Backend** (`ggml_backend_i`) | `graph_compute()`, `synchronize()`, `graph_optimize()` | 실제 연산 실행 |
| **Buffer Type** (`ggml_backend_buffer_type_i`) | `alloc_buffer()`, `get_alignment()`, `is_host()` | 메모리 할당 전략 |

---

## 1.2 백엔드 등록 메커니즘

### 정적 등록 (빌드타임)

[ggml/src/ggml-backend-reg.cpp](../../ggml/src/ggml-backend-reg.cpp) 에서 컴파일 옵션에 따라 백엔드를 정적으로 등록합니다:

```cpp
// ggml-backend-reg.cpp 내부
#ifdef GGML_USE_HEXAGON
    ggml_backend_hexagon_reg,   // Hexagon 백엔드 등록 함수 포인터
#endif
```

### 동적 등록 (런타임)

`GGML_BACKEND_DL_IMPL` 매크로를 통해 `.so` / `.dll` 동적 라이브러리로 빌드 후 런타임에 로딩할 수도 있습니다.

Hexagon 백엔드는 [ggml-hexagon.cpp의 마지막 줄](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3301)에서 이 매크로를 선언합니다:

```cpp
GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
```

---

## 1.3 백엔드 스케줄러 (`ggml_backend_sched`)

스케줄러는 하나의 계산 그래프를 여러 백엔드에 걸쳐 실행하는 핵심 컴포넌트입니다.

### 핵심 함수: `ggml_backend_sched_split_graph()`

위치: [ggml/src/ggml-backend.cpp](../../ggml/src/ggml-backend.cpp)

이 함수는 **5 패스**에 걸쳐 그래프를 분할합니다:

| 패스 | 목적 |
|------|------|
| **패스 1** | 이미 할당된 버퍼를 기준으로 백엔드 ID를 부여 |
| **패스 2** | GPU 백엔드를 인접 노드로 확장 (아래→위→아래→위 4방향) |
| **패스 3** | 호환 버퍼 타입의 더 높은 우선순위 백엔드로 업그레이드 |
| **패스 4** | 나머지 미할당 노드에 source/view 기반 백엔드 부여 |
| **패스 5** | 백엔드 경계에서 Split을 만들고, 크로스-백엔드 입력에 대해 복사 텐서 생성 |

### 분할 결과 구조

```
Split #0: [Hexagon] nodes[0..47]       ← MUL_MAT, ADD, RMS_NORM 등
  inputs: [tensor_A from CPU]          ← CPU→Hexagon 복사 필요
Split #1: [CPU]     nodes[48..50]      ← Hexagon이 미지원하는 Op
Split #2: [Hexagon] nodes[51..99]      ← 다시 Hexagon으로 복귀
```

### Op 지원 여부 판단

스케줄러는 각 백엔드의 `supports_op()` 함수를 호출하여 해당 연산을 실행 가능한지 판단합니다:

```cpp
// ggml-backend.cpp 내부
if (ggml_backend_supports_op(sched->backends[b], node)) {
    // 이 백엔드가 이 Op을 처리할 수 있음
}
```

Hexagon 백엔드에서 `false`를 반환하면, 스케줄러는 자동으로 CPU 폴백을 선택합니다.

---

## 1.4 그래프 실행 흐름

```
ggml_backend_sched_graph_compute(sched, graph)
  ├── ggml_backend_sched_split_graph()     // 분할
  ├── ggml_backend_sched_alloc_splits()    // 메모리 할당
  └── for each split:
       ├── copy inputs (CPU↔Hexagon)       // 크로스-백엔드 텐서 복사
       ├── backend->graph_optimize(split)  // 백엔드별 그래프 최적화
       └── backend->graph_compute(split)   // 실제 연산 실행
```

---

## 1.5 Hexagon 백엔드가 제공하는 인터페이스 구현

Hexagon 백엔드는 다음 인터페이스를 구현합니다 (위치: [ggml-hexagon.cpp](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)):

```cpp
static struct ggml_backend_i hexagon_backend_i = {
    .get_name       = ggml_backend_hexagon_name,
    .free           = ggml_backend_hexagon_free,
    .synchronize    = ggml_backend_hexagon_synchronize,
    .graph_compute  = ggml_backend_hexagon_graph_compute,   // ★ 핵심
    .graph_optimize = ggml_backend_hexagon_graph_optimize,  // ★ 그래프 최적화
};
```

특이사항:
- `set_tensor_async`, `get_tensor_async`: 구현하지 않음 (NULL). 호스트 버퍼를 직접 사용하므로 비동기 전송이 불필요.
- `event_record/wait`: 미구현. dspqueue의 동기화 메커니즘을 직접 사용.
- `graph_optimize`: 구현함. Op 퓨전과 MUL_MAT 재정렬 최적화를 수행.
