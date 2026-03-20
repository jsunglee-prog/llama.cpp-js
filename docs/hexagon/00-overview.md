# llama.cpp Hexagon DSP 백엔드 분석 보고서

## 목차

| 문서 | 내용 |
|------|------|
| [00-overview.md](00-overview.md) | 전체 프로젝트 구조 및 Hexagon 모듈 위치 |
| [01-backend-abstraction.md](01-backend-abstraction.md) | ggml 하드웨어 가속 추상화 방식 |
| [02-hexagon-files.md](02-hexagon-files.md) | Hexagon 소스 코드 경로 및 주요 파일 |
| [03-dispatch-pipeline.md](03-dispatch-pipeline.md) | llama-cli → Hexagon DSP 오프로딩 과정 |
| [04-memory-and-weights.md](04-memory-and-weights.md) | 가중치 배치 및 메모리 관리 |
| [05-graph-splitting.md](05-graph-splitting.md) | 그래프 빌드 및 백엔드 분할 |
| [06-vtcm-tiling-dma.md](06-vtcm-tiling-dma.md) | VTCM 타일링 및 DMA 전송 메커니즘 |
| [07-result-sync.md](07-result-sync.md) | 연산 결과 회수 및 CPU 동기화 |
| [08-code-level-deep-dive.md](08-code-level-deep-dive.md) | 코드 레벨 심층 분석 (함수명, 라인, 상수, 데이터 흐름) |

---

## 0. 전체 프로젝트 폴더 구조와 주요 모듈의 역할

### 프로젝트 최상위 구조

```
llama.cpp/
├── src/                    # llama.cpp 핵심 라이브러리 (모델 로딩, 그래프 빌드, 토큰 생성)
├── common/                 # CLI 공용 유틸리티 (arg parsing, sampling, chat 등)
├── examples/               # llama-cli, llama-server 등 실행파일
├── include/                # 퍼블릭 헤더 (llama.h)
├── ggml/                   # 텐서 연산 라이브러리 (백엔드 추상화 포함)
│   ├── include/            # ggml.h, ggml-backend.h 등 퍼블릭 API
│   └── src/                # 백엔드 구현체들
│       ├── ggml.c/cpp      # 코어 텐서 연산 + 그래프 엔진
│       ├── ggml-backend.cpp # 백엔드 스케줄러 (ggml_backend_sched)
│       ├── ggml-cpu/       # CPU 백엔드 (NEON, AVX, etc.)
│       ├── ggml-cuda/      # NVIDIA CUDA 백엔드
│       ├── ggml-metal/     # Apple Metal 백엔드
│       ├── ggml-vulkan/    # Vulkan 백엔드
│       ├── ggml-hexagon/   # ★ Qualcomm Hexagon DSP 백엔드
│       ├── ggml-opencl/    # OpenCL 백엔드
│       ├── ggml-sycl/      # Intel oneAPI SYCL 백엔드
│       ├── ggml-hip/       # AMD ROCm 백엔드
│       ├── ggml-cann/      # Huawei Ascend 백엔드
│       ├── ggml-rpc/       # 원격 프로시저 호출 백엔드
│       └── ...             # 기타 (blas, musa, webgpu 등)
├── gguf-py/                # GGUF 포맷 Python 도구
├── convert_hf_to_gguf.py   # HuggingFace → GGUF 변환기
├── tests/                  # 테스트 스위트
└── tools/                  # quantize, gguf-hash 등 도구
```

### 주요 모듈 역할

| 모듈 | 역할 |
|------|------|
| `src/` | `llama_model_load()`, `llama_graph_build()`, `llama_decode()` 등 모델 추론 핵심 로직 |
| `ggml/src/ggml.c` | 텐서 자료구조, 계산 그래프(cgraph) 정의, 연산자(Op) 정의 |
| `ggml/src/ggml-backend.cpp` | `ggml_backend_sched` - 멀티 백엔드 그래프 분할 및 실행 스케줄러 |
| `ggml/src/ggml-backend-reg.cpp` | 백엔드 레지스트리 - 동적 라이브러리 로딩 및 백엔드 등록 |
| `ggml/src/ggml-hexagon/` | Hexagon DSP 백엔드 - CPU↔DSP 통신, 텐서 repack, 그래프 최적화 |
| `common/` | CLI 인터페이스, 샘플링, 채팅 파서 등 애플리케이션 레벨 유틸리티 |

### 데이터 흐름 요약

```
사용자 입력 → llama-cli (examples/)
  → llama_decode() (src/)
    → llama_graph_build() → ggml_cgraph 생성
    → ggml_backend_sched_graph_compute()
      → ggml_backend_sched_split_graph() → CPU/Hexagon 분할
      → 각 백엔드별 graph_compute() 호출
        → Hexagon: dspqueue를 통해 DSP로 Op 전송
          → DSP에서 HVX/VTCM으로 연산 수행
          → 결과를 dspqueue response로 CPU에 통보
    → 로짓(Logits) 반환 → 토큰 샘플링 → 출력
```
