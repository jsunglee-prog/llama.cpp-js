# htp-ops-lib 분석 문서 목차

> Qualcomm Hexagon NPU(cDSP)용 LLM 추론 커스텀 연산자 라이브러리 분석  
> 참조 논문: [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324)

---

## 문서 목록

| # | 문서 | 설명 |
|---|------|------|
| 0 | [프로젝트 구조](00_프로젝트_구조.md) | 폴더 구조, 빌드 산출물, 모듈별 역할, 데이터 흐름 개요, 지원 연산 목록 |
| 1 | [HMX 하드웨어 가속 추상화](01_HMX_하드웨어_가속_추상화.md) | HMX 타일 인라인 어셈블리, 자원 관리, 스핀 락, 연산자별 사용 패턴 |
| 2 | [VTCM → HMX 데이터 흐름](02_VTCM_HMX_데이터_흐름.md) | 메모리 계층, DMA 파이프라인, MatMul/FlashAttention 각각의 DDR→VTCM→HMX 경로 |
| 3 | [HMX Repack(Crouton) 방식](03_HMX_Repack_방식.md) | Crouton 레이아웃 상세, 인덱싱 공식, FP32→FP16 변환, 양자화 역변환, 크기 분석 |
| 4 | [HVX vs HMX 병렬 처리 비교](04_HVX_HMX_병렬처리_비교.md) | HVX/HMX 스펙 비교, 연산별 역할 분담, FFN/Attention 이점 분석, 벤치마크 |
| 5 | [VTCM 8MB 타일링 전략](05_VTCM_타일링_전략.md) | VTCM 할당 아키텍처, MatMul 4-영역 분할, FlashAttention 동적 Br/Bc, exp2 테이블 상주 |
| 6 | [Git 커밋 분석](06_Git_커밋_분석.md) | Git 상태 확인 결과 (untracked 폴더, 커밋 이력 없음) |
| 7 | [단독 테스트 및 크로스빌드](07_단독_테스트_및_크로스빌드.md) | Stub/Skel 독립 빌드, 테스트 프로그램, Hexagon SDK 크로스빌드, 디바이스 배포 |
| 8 | [ggml-hexagon 통합 분석](08_ggml_hexagon_통합_분석.md) | 기존 ggml-hexagon과의 아키텍처 비교, 통합 방안 3가지 |
| 9 | [llama.cpp-npu 포크 통합 분석](09_llama_cpp_npu_포크_통합_분석.md) | 포크의 실제 통합 방식: dlopen, 프로토콜 헤더 복사, 공유 메모리 폴링, 하이브리드 실행 |
| 10 | [ggml-hexagon HMX 통합 작업 계획](10_ggml_hexagon_HMX_통합_작업_계획.md) | 기존 skel에 HMX 커널 병합: 5단계 작업 계획, 수정 파일 체크리스트, VTCM 통합, Crouton repack |

---

## 핵심 아키텍처 요약

```
Host (Android CPU)                    DSP (Hexagon cDSP)
┌──────────────┐    FastRPC/         ┌──────────────────────────────┐
│ session.c    │    SharedMem        │ 자원 관리                     │
│ op_export.c  │ ◀═══════════▶      │  power.c / hmx_mgr.c         │
│ test.c       │                     │  vtcm_mgr.cc / worker_pool.c │
└──────────────┘                     ├──────────────────────────────┤
                                     │ 통신                         │
                                     │  commu.c / op_executor.cc    │
                                     ├──────────────────────────────┤
                                     │ 연산자 (HMX + HVX + DMA)    │
                                     │  mat_mul.c (FFN 행렬곱)     │
                                     │  flash_attn.c (어텐션)      │
                                     │  rms_norm.c (정규화)        │
                                     │  precompute_table.c (exp2)  │
                                     └──────────────────────────────┘
```

## 분석 환경

- **분석 일시**: 2025년 6월
- **대상 코드**: `llama.cpp/htp-ops-lib/` (untracked)
- **소스 파일 수**: 헤더 18개 + 소스 15개 = 총 33개 파일
- **총 코드 라인**: 약 9,000+ 줄 (주석/공백 포함)
