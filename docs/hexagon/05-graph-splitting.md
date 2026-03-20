# 5. 그래프 빌드 및 백엔드 분할(Splitting) 분석

> 분석 대상 코드: [ggml-backend.cpp:920-1300](../../ggml/src/ggml-backend.cpp) (스케줄러), [ggml-hexagon.cpp:2700-2850](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp) (graph_optimize), [ggml-hexagon.cpp:2984-3100](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp) (supports_op)

---

## 5.1 그래프 분할 메커니즘: `ggml_backend_sched_split_graph()`

### 5-패스 분할 알고리즘

위치: [ggml-backend.cpp](../../ggml/src/ggml-backend.cpp)

#### 패스 1: 초기 할당 (버퍼 기반)

```cpp
// 리프 노드와 연산 노드의 기존 버퍼에서 백엔드 ID를 추론
for (int i = 0; i < graph->n_nodes; i++) {
    node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
    // → 텐서의 buffer가 Hexagon이면 → backend_id = Hexagon
}
```

**핵심**: 가중치가 Hexagon repack 버퍼에 할당되어 있으면, 해당 가중치를 사용하는 MUL_MAT 노드가 자동으로 Hexagon에 할당됩니다.

#### 패스 2: 확장 (4방향 인접 확장)

```
[GPU 아래로 확장] → [GPU 위로 확장] → [나머지 아래로 확장] → [나머지 위로 확장]
```

```cpp
// GPU(= non-CPU, non-lowest-priority) 백엔드를 인접 노드로 확장
// Hexagon이 지원하는 Op이면 Hexagon으로 할당
int cur_backend_id = -1;
for (int i = 0; i < graph->n_nodes; i++) {
    if (*node_backend_id != -1) {
        if (*node_backend_id == sched->n_backends - 1) {
            cur_backend_id = -1;  // CPU는 건너뜀
        } else {
            cur_backend_id = *node_backend_id;  // Hexagon 등
        }
    } else if (cur_backend_id != -1) {
        // Hexagon이 이 Op을 지원하면 → Hexagon으로 할당
        ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
    }
}
```

**결과**: Hexagon MUL_MAT 사이에 있는 ADD, RMS_NORM, SCALE 등이 Hexagon이 지원하면 같이 묶입니다.

#### 패스 3: 업그레이드 + 폴백 할당

```cpp
// 미할당 노드: 가장 많은 호환 입력을 가진 백엔드 선택
// 이미 할당된 노드: 같은 buft의 더 높은 우선순위 백엔드로 승격
for (int b = 0; b < sched->n_backends; b++) {
    if (ggml_backend_supports_op(sched->backends[b], node)) {
        // 호환 입력 수 카운트 → 최대치 선택
    }
}
```

#### 패스 4: 나머지 입력 텐서에 백엔드 할당

```cpp
// view_src, 현재 노드의 백엔드에서 src 백엔드 추론
```

#### 패스 5: 실제 분할 (Split) 생성

```cpp
for (int i = 0; i < graph->n_nodes; i++) {
    if (node_backend_id != cur_backend_id || need_new_split) {
        // 새로운 Split 생성
        split->i_end = i;
        split = &sched->splits[++i_split];
        split->backend_id = node_backend_id;
        split->i_start = i;
    }

    // 크로스-백엔드 입력 텐서에 대해 복사 텐서 생성
    if (src_backend_id != cur_backend_id && !buffer_supported) {
        // ggml_dup_tensor_layout()으로 복사본 생성
        // 실행 시 자동으로 데이터 복사됨
    }
}
```

---

## 5.2 Hexagon 백엔드의 `supports_op()` 판단 기준

위치: [ggml-hexagon.cpp:2984](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)

### 2단계 검증 구조

```
supports_op(op):
  │
  ├── 1단계: 버퍼 호환성
  │     └── ggml_hexagon_supported_buffers(sess, op)
  │           → 모든 src와 dst가 이 세션의 Hexagon 버퍼인지 확인
  │           → 다른 세션이나 CPU 버퍼가 섞여있으면 false
  │
  └── 2단계: Op별 세부 검증
        └── ggml_hexagon_supported_mul_mat(sess, op) 등
```

### Op별 지원 조건 상세

#### MUL_MAT

```cpp
static bool ggml_hexagon_supported_mul_mat(const ggml_hexagon_session * sess, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // src0 타입: Q4_0, Q8_0, MXFP4, F16 만 지원
    const ggml_type src0_type = src0->type;
    if (src0_type != Q4_0 && src0_type != Q8_0 && src0_type != MXFP4 && src0_type != F16)
        return false;

    // src1 타입: F32 또는 F16만 지원
    if (src1->type != F32 && src1->type != F16) return false;

    // dst 타입: F32만 지원
    if (dst->type != F32) return false;

    // 양자화 타입은 repack 버퍼 필수
    if (ggml_is_quantized(src0_type)) {
        if (!src0->buffer || !ggml_backend_buffer_is_hexagon_repack(src0->buffer))
            return false;
    }
}
```

#### SOFTMAX

```cpp
static bool ggml_hexagon_supported_softmax(...) {
    // src0: F32만 지원
    // src1(mask): F16 또는 F32
    // dst: F32만 지원
    // opmask 확인
}
```

#### FLASH_ATTN_EXT

```cpp
static bool ggml_hexagon_supported_flash_attn_ext(...) {
    // Q: F16 또는 F32
    // K, V: F16만 지원
    // dst: F16 또는 F32
    // logit softcap: 지원
}
```

#### 패스스루 Op (무조건 지원)

```
GGML_OP_NONE, RESHAPE, VIEW, PERMUTE, TRANSPOSE → 항상 true
```

이들은 메타데이터만 변경하고 실제 연산이 없으므로, DSP로 전송하지 않고 즉시 통과합니다.

### Op 거부(Reject) → CPU 폴백 시나리오

| 거부 사유 | 예시 |
|----------|------|
| 미지원 데이터 타입 | MUL_MAT에서 src0이 Q5_K → Hexagon 미지원 |
| 버퍼 불일치 | src가 다른 세션의 버퍼에 있음 |
| 미지원 Op | NORM, CONCAT 등 → Hexagon에서 구현되지 않음 |
| opmask 필터 | `GGML_HEXAGON_OPMASK` 환경변수로 특정 Op 비활성화 |
| Repack 미적용 | 양자화 가중치에 repack 버퍼가 아닌 일반 버퍼 사용 |

---

## 5.3 Hexagon 그래프 최적화: `graph_optimize()`

위치: [ggml-hexagon.cpp:2740](../../ggml/src/ggml-hexagon/ggml-hexagon.cpp)

Hexagon 백엔드는 스케줄러의 그래프 분할 **이후**, 각 서브그래프에 대해 추가 최적화를 수행합니다.

### 5.3.1 Op 퓨전 (Fusion)

```
입력 그래프:      [RMS_NORM] → [MUL] → [ADD] → [MUL_MAT]
최적화 후:        [RMS_NORM+MUL+ADD] → [MUL_MAT]
                   (하나의 fused node)
```

퓨전 대상:
- `ADD`, `MUL`, `NORM`, `RMS_NORM` 연속 체인
- `ggml_can_fuse()` 함수로 퓨전 가능 여부 판단
- 최대 `MAX_FUSE=16`개까지 연결 가능

### 5.3.2 MUL_MAT 재정렬 (Reorder for VTCM Reuse)

**목적**: 같은 src1(입력)을 공유하는 MUL_MAT Op들을 연속 배치하여, VTCM에 양자화된 src1을 한 번만 올리고 재사용합니다.

```cpp
static std::vector<int> ggml_hexagon_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    for (int i0 = 0; i0 < n; i0++) {
        if (node0.stackable()) {
            // 앞으로 16개 노드까지 검색
            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (node1.stackable() && node1.same_input(node0)) {
                    // 같은 src1을 가진 MUL_MAT → 바로 뒤에 배치
                    res.push_back(i1);
                    used[i1] = true;
                }
            }
        }
    }
}
```

실행 시 효과 (`graph_compute`에서):
```cpp
// SKIP_QUANTIZE 플래그 자동 적용
if (op_reuse_src1(node, prev_op)) {
    flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
    // → DSP에서 src1 양자화를 건너뛰고 VTCM 데이터 재사용
}
```

---

## 5.4 CPU↔Hexagon 간 데이터 의존성 동기화

### Split 경계에서의 자동 복사

스케줄러의 패스 5에서 생성된 "복사 텐서"는 Split 실행 전에 자동으로 데이터를 이동합니다:

```
Split #0 (Hexagon):  [MUL_MAT → ADD → RMS_NORM → ...]
                     ↓ 출력 텐서가 Hexagon 버퍼에 있음
Split #1 (CPU):      [NORM (Hexagon 미지원)]
                     ↑ Hexagon 출력을 CPU 복사 텐서로 복사
                     ↓ CPU 결과가 CPU 버퍼에 있음
Split #2 (Hexagon):  [MUL_MAT → ...]
                     ↑ CPU 결과를 Hexagon 복사 텐서로 복사
```

### 호스트 버퍼 모드 (`opt_hostbuf=1`, 기본값)

호스트 버퍼 모드에서는 Hexagon의 regular 버퍼가 `is_host=true`로 선언됩니다:
- **의미**: CPU도 이 버퍼를 직접 읽을 수 있음
- **효과**: 많은 경우 별도의 복사 없이 CPU가 Hexagon 버퍼의 데이터를 직접 사용 가능
- **제한**: repack 버퍼는 여전히 `is_host=false` (HVX 레이아웃이라 CPU가 직접 읽을 수 없음)

### 동기화 시점

```
graph_compute() 종료 시:
  sess->flush()
    └── dspqueue_read() 반복
          → 모든 op_pending이 0이 될 때까지 응답 대기
          → 이 시점에서 모든 DSP 연산 결과가 DRAM에 확정

다음 Split 시작 전:
  → 이전 Split의 synchronize() 호출
  → 크로스-백엔드 복사 텐서에 대해 memcpy 또는 buffer copy 실행
  → 다음 백엔드의 graph_compute() 시작
```
