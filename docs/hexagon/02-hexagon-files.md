# 2. Hexagon DSP 백엔드 소스 코드 경로 및 주요 파일

---

## 2.1 디렉토리 구조

```
ggml/src/ggml-hexagon/
├── ggml-hexagon.cpp          # ★ CPU측 백엔드 구현 (3301줄)
├── htp-drv.cpp               # FastRPC 드라이버 로더 (419줄)
├── htp-drv.h                 # 드라이버 API 헤더
├── op-desc.h                 # Op 디스크립터 헤더
├── libdl.h                   # 동적 라이브러리 로딩 래퍼
├── libggml-htp.inf           # DSP 모듈 인포 파일
├── CMakeLists.txt            # 빌드 설정
│
└── htp/                      # ★ DSP측 구현 (Hexagon DSP에서 실행되는 코드)
    ├── main.c                # DSP메인 루프, 메시지 디스패치, VTCM 관리 (1200줄)
    ├── matmul-ops.c          # 행렬곱 커널 (2572줄) - 가장 큰 파일
    ├── flash-attn-ops.c      # Flash Attention 커널 (714줄)
    ├── softmax-ops.c         # Softmax 커널 (420줄)
    ├── binary-ops.c          # 이항 연산 (ADD, MUL, SUB, DIV)
    ├── unary-ops.c           # 단항 연산 (RMS_NORM, SCALE, SQR, SQRT)
    ├── act-ops.c             # 활성화 함수 (SILU, GELU, SWIGLU, GEGLU)
    ├── rope-ops.c            # RoPE (Rotary Position Embedding)
    ├── cpy-ops.c             # 텐서 복사 연산
    ├── set-rows-ops.c        # SET_ROWS 연산
    ├── get-rows-ops.c        # GET_ROWS 연산
    ├── argsort-ops.c         # 정렬 연산
    ├── sum-rows-ops.c        # 행 합산 연산
    ├── ssm-conv.c            # SSM 컨볼루션 (Mamba 모델)
    │
    ├── worker-pool.c         # QuRT 스레드 풀 (294줄)
    ├── worker-pool.h         # 워커 풀 API
    ├── hex-dma.c             # DMA 큐 관리
    ├── hex-dma.h             # DMA API
    │
    ├── htp-msg.h             # CPU↔DSP 메시지 프로토콜 정의
    ├── htp-ops.h             # Op 컨텍스트 및 함수 선언
    ├── htp-ctx.h             # DSP 컨텍스트 구조체
    ├── hvx-types.h           # HVX 벡터 타입 정의
    │
    ├── hvx-arith.h           # HVX 산술 유틸리티
    ├── hvx-base.h            # HVX 기본 연산
    ├── hvx-copy.h            # HVX 복사 연산
    ├── hvx-div.h             # HVX 나눗셈
    ├── hvx-exp.h             # HVX 지수 함수
    ├── hvx-floor.h           # HVX floor 연산
    ├── hvx-inverse.h         # HVX 역수 연산  
    ├── hvx-reduce.h          # HVX 리덕션 연산
    ├── hvx-scale.h           # HVX 스케일 연산
    ├── hvx-sigmoid.h         # HVX 시그모이드
    ├── hvx-sqrt.h            # HVX 제곱근
    ├── hvx-utils.h           # HVX 유틸리티
    ├── hvx-dump.h            # HVX 디버그 덤프
    ├── hex-fastdiv.h         # 빠른 정수 나눗셈 유틸리티
    ├── hex-dump.h            # 디버그 덤프
    ├── hex-utils.h           # 범용 유틸리티
    │
    ├── htp_iface.idl         # FastRPC IDL 인터페이스 정의
    ├── CMakeLists.txt        # DSP측 빌드 설정
    └── cmake-toolchain.cmake # Hexagon 크로스컴파일 툴체인
```

---

## 2.2 주요 파일별 상세 역할

### CPU측 코드 (호스트 프로세서에서 실행)

| 파일 | 줄 수 | 핵심 역할 |
|------|-------|---------|
| **ggml-hexagon.cpp** | 3301 | 백엔드 전체 구현: 세션 관리, 버퍼 할당, Op dispatch, 그래프 최적화, supports_op, repack 함수 |
| **htp-drv.cpp** | 419 | `libcdsprpc.so`/`.dll` 동적 로딩, FastRPC 함수 포인터 바인딩, DSP 아키텍처 버전 쿼리 |

### DSP측 코드 (Hexagon DSP에서 실행)

| 파일 | 줄 수 | 핵심 역할 |
|------|-------|---------|
| **main.c** | 1200 | DSP 메인 루프, 패킷 콜백, VTCM 할당/해제, HAP 전원 관리, DMA 큐 초기화 |
| **matmul-ops.c** | 2572 | q4x4x2/q8x4x2/mxfp4x4x2 내적 커널, matmul_2d/4d 타일링, DMA 파이프라이닝 |
| **flash-attn-ops.c** | 714 | Flash Attention: Q/K/V DMA 프리페치, 블록 단위 online softmax |
| **softmax-ops.c** | 420 | HVX 최적화 softmax: max-find → exp → sum → normalize |
| **worker-pool.c** | 294 | QuRT 스레드 풀: futex 동기화, 원자적 작업 분배 |
| **hex-dma.c** | - | DMA 큐: DDR↔VTCM 비동기 메모리 전송 관리 |

### 프로토콜 / 자료구조 헤더

| 파일 | 핵심 정의 |
|------|---------|
| **htp-msg.h** | `htp_general_req` (64B Op params + 6 텐서), `htp_general_rsp` (상태 + 프로파일링), `htp_op` enum (25종) |
| **htp-ops.h** | `htp_ops_context` (Op 실행 컨텍스트), `htp_spad` (VTCM 스크래치패드), Op 함수 선언 |
| **htp-ctx.h** | `htp_context` (dspqueue, DMA큐[스레드별], 워커풀, VTCM 메타, opmask) |
| **hvx-types.h** | `VLEN=128`, HVX 벡터 union 타입, x2/x4/x8 벡터 구조체 |

---

## 2.3 빌드 아키텍처

Hexagon 백엔드는 **두 개의 별도 바이너리**로 빌드됩니다:

```
[호스트 빌드 (x86/ARM)]
  ggml-hexagon.cpp + htp-drv.cpp
  → libggml-hexagon.so (호스트 공유 라이브러리)

[DSP 크로스 빌드 (Hexagon)]
  htp/main.c + htp/*.c
  → libggml-htp-v{arch}.so (DSP 공유 라이브러리, v68/v73/v75/v79/v81)
```

DSP 라이브러리는 `htp_iface.idl` 에서 정의한 FastRPC 스켈레톤 인터페이스를 구현하며, 호스트에서 `remote_handle64_open()`으로 이 `.so`를 DSP에 로딩합니다.
