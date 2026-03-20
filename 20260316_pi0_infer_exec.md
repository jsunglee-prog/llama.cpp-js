# pi0\_infer Hexagon 실행 파일 구성 정리

> 관련 코드: [examples/pi0_infer/pi0_infer.cpp](examples/pi0_infer/pi0_infer.cpp), [examples/pi0_infer/CMakeLists.txt](examples/pi0_infer/CMakeLists.txt), [examples/CMakeLists.txt](examples/CMakeLists.txt)

---

## 1. 프로젝트 구조 갱신 내역
- **새 예제 추가**: `examples/pi0_infer/` 폴더에 `pi0_infer.cpp`와 독립적인 `CMakeLists.txt`를 배치하여 Hexagon용 단독 추론 실행 파일을 제공.
- **빌드 트리 연동**: 상위 [examples/CMakeLists.txt](examples/CMakeLists.txt)에 `add_subdirectory(pi0_infer)`를 삽입해 전체 CMake 흐름에서 자동으로 타깃이 생성되도록 구성.

## 2. 실행 파일 설계 개요
### 2.1 모델/실행 파라미터
- `struct pi0_config`는 시퀀스 길이(784), 임베딩 차원(2048), GQA 비율(8:1), 레이어 수(18), hidden multiplier(×8) 등 요구 조건을 기본값으로 담고 `finalize()` 단계에서 `n_head`, `n_head_kv`, `hidden_dim`을 파생 계산.
- CLI 인자로 `--layers`, `--seq`, `--dim`, `--gqa`, `--head-dim`, `--hidden-mult`, `--arena-mb`, `--seed`를 지원하므로 다른 SoC나 실험 구성을 쉽게 주입 가능.

### 2.2 레이어 그래프와 Hexagon 백엔드 흐름
- 매 레이어마다 `layer_weights`가 Wq/Wk/Wv/Wo/MLP up/down 행렬을 F16 랜덤값으로 생성해 VTCM에 올릴 준비를 수행.
- `build_attention()`은 Q/K/V를 각각 `ggml_mul_mat`으로 생성 후 `[head_dim, seq_len, head]` 형태로 리쉐이프하여 `ggml_flash_attn_ext`(GQA/Flash Attention)를 호출, Hexagon Flash 커널이 K/V 64행 타일링을 자동 적용하도록 했음.
- `build_mlp()`는 `ggml_silu` + 두 개의 선형 변환을 통해 `2048 → 16384 → 2048` 경로를 만들고, Residual을 통해 입력과 합산.
- `run_layer()`는 `ggml_backend_graph_copy()`로 전체 그래프를 Hexagon 백엔드 메모리(DDR 공유 영역)로 이관한 후 한 번의 RPC 호출로 실행, 종료 시 `ggml_backend_tensor_get()`으로 결과를 DDR→HLOS로 복사한다. 이 과정에서 Hexagon 커널(`matmul_2d`, `flash_attn_ext`)이 자체적으로 src0/src1/dst를 VTCM으로 스트리밍하므로 DDR DMA 오버헤드가 최소화된다.

### 2.3 실행 흐름 요약
1. `ggml_backend_load_all()` → `ggml_backend_hexagon_init()`로 DSP 세션을 확보.
2. 초기 hidden state(784×2048)를 F16 랜덤값으로 채움.
3. 각 레이어마다 그래프를 새로 구성하여 `ggml_backend_graph_compute()`로 수행, 출력은 다음 레이어의 입력으로 사용.
4. 전체 18개 레이어가 끝나면 RMS 값과 샘플 출력을 표준 출력에 기록.

## 3. 빌드 및 실행 절차
1. **CMake 설정**
   ```bash
   cmake -B build -DLLAMA_CUBLAS=off -DLLAMA_ACCELERATE=off
   ```
   (Hexagon 백엔드가 켜진 툴체인을 사용해야 하며, DSP RPC 라이브러리가 링크 가능한 환경이어야 함.)
2. **타깃 빌드**
   ```bash
   cmake --build build --target pi0_infer -j
   ```
3. **실행 예시**
   ```bash
   ./build/bin/pi0_infer \
       --layers 18 --seq 784 --dim 2048 \
       --gqa 8 --head-dim 64 --hidden-mult 8 \
       --arena-mb 256 --seed 1234
   ```
   - DSP 상주 메모리(VTCM)는 커널 단에서 자동 관리되며, `--arena-mb`는 HLOS 측 그래프 메타데이터/임시 버퍼 크기만 제어한다.

## 4. Hexagon/VTCM 최적화 포인트
- MatMul 경로는 `ggml_backend_graph_copy()` → Hexagon `op_matmul`로 연결되어, src1이 8MB VTCM을 넘으면 자동으로 DMA 스트리밍·타일링(`matmul_2d`, `matmul_4d`)을 수행한다.
- Flash Attention은 K/V를 64행 단위로 더블 버퍼링하므로 시퀀스 길이에 따라 RPC 호출 수가 증가하지 않는다.
- 가중치는 레이어 단위로 재생성/해제되므로 한 번에 VTCM에 올라가는 크기는 단일 레이어에 한정되고, HLOS→DSP 전송 또한 레이어별 단일 그래프로 묶여 호출 횟수가 최소화된다.

## 5. 향후 확장 아이디어
1. **MatMul src1 타일링 노출**: 현재 Hexagon 커널 내부의 자동 타일링을 신뢰하지만, `pi0_infer`에서 직접 청크를 정의해 VTCM 예산을 시각화하면 메모리/대역폭 추적이 용이해진다.
2. **KV 캐시 연동**: 실 모델에서는 KV 캐시를 유지해야 하므로, `state` 벡터 외에 KV 버퍼를 Hexagon 백엔드 버퍼로 고정시키는 예제가 유용하다.
3. **프로파일 뷰어 훅**: `ggml_backend_tensor_get()` 직후 성능 카운터(FASTRPC 프로파일) 출력 기능을 추가하면 SoC별 튜닝 시간이 단축된다.
