# 6. VTCM 타일링 및 DMA 전송 메커니즘

> 분석 대상 코드: [htp/main.c](../../ggml/src/ggml-hexagon/htp/main.c) (VTCM 관리), [htp/matmul-ops.c](../../ggml/src/ggml-hexagon/htp/matmul-ops.c) (타일링 + DMA), [htp/flash-attn-ops.c](../../ggml/src/ggml-hexagon/htp/flash-attn-ops.c) (Flash Attention DMA), [htp/hex-dma.h](../../ggml/src/ggml-hexagon/htp/hex-dma.h)

---

## 6.1 VTCM (Vector Tightly Coupled Memory) 관리

### VTCM 할당

위치: [htp/main.c](../../ggml/src/ggml-hexagon/htp/main.c)

```c
static int vtcm_alloc(struct htp_context * ctx) {
    // VTCM 가용 크기 조회
    unsigned int vtcm_max_page, vtcm_total, vtcm_avail;
    HAP_query_vtcm_page(&vtcm_max_page, &vtcm_total, &vtcm_avail);
    
    ctx->vtcm_size = vtcm_avail;  // 기본 ~8MB

    // Compute Resource 속성 설정
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param(
        &attr, 
        ctx->vtcm_size, 
        1  // cached = 1 (캐시 모드)
    );
    HAP_compute_res_attr_set_hmx(&attr, 1);  // HMX 활성화
    HAP_compute_res_attr_set_release_callback(
        &attr, vtcm_release_callback, ctx
    );

    // VTCM 할당 요청
    ctx->vtcm_rctx = HAP_compute_res_acquire(&attr, 100 /*timeout ms*/);
    ctx->vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
}
```

### VTCM 경쟁 관리: 지연 획득(Lazy Acquire) 패턴

```c
static void vtcm_acquire(struct htp_context * ctx) {
    if (!atomic_load(&ctx->vtcm_valid)) {
        // VTCM이 다른 세션에게 빼앗겼을 수 있음
        // 우선순위를 올려서 다시 획득 시도
        
        // 1. 현재 스레드 우선순위 저장
        int prev_prio = get_thread_priority();
        
        // 2. 임시로 최고 우선순위로 올림
        set_thread_priority(highest_priority);
        
        // 3. Re-acquire
        ctx->vtcm_rctx = HAP_compute_res_acquire(&attr, timeout);
        
        // 4. 원래 우선순위 복원
        set_thread_priority(prev_prio);
        
        ctx->vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
        atomic_store(&ctx->vtcm_valid, true);
    }
    
    atomic_store(&ctx->vtcm_inuse, true);
}

static void vtcm_release(struct htp_context * ctx) {
    atomic_store(&ctx->vtcm_inuse, false);
    
    // 해제 콜백에서 요청이 오면 실제 해제
    if (atomic_load(&ctx->vtcm_needs_release)) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        atomic_store(&ctx->vtcm_valid, false);
    }
}
```

**Release Callback**: 다른 DSP 세션이 VTCM을 요청하면 콜백이 호출됩니다. 현재 Op 실행 중이면 (`vtcm_inuse=true`) 해제를 지연하고, Op 완료 후 해제합니다.

---

## 6.2 MatMul 타일링 전략

### VTCM 스크래치패드 레이아웃

```
VTCM (8MB)
├── src0_spad (스레드별)  : 16행 × row_size_padded × n_threads
├── src1_spad (공유)      : 전체 src1 행 (양자화된 입력)
└── dst_spad (스레드별)   : 2행 × dst_row_size × n_threads
```

타일링 상수:
```c
#define MM_SPAD_SRC0_NROWS  16   // VTCM에 한 번에 올리는 src0 행 수
#define MM_SPAD_SRC1_NROWS  16   // 4D 경로에서의 src1 행 수 
#define MM_SPAD_DST_NROWS    2   // VTCM에서 유지하는 dst 행 수
```

### matmul_2d: 최적화된 2D 행렬곱 (주 경로)

위치: [htp/matmul-ops.c:1569](../../ggml/src/ggml-hexagon/htp/matmul-ops.c)

**전제 조건**: src1(입력)이 VTCM에 완전히 로딩되어 있어야 함

```
작업 흐름 (스레드별):

1. 초기 프리페치 (Prefill):
   DMA: DRAM → VTCM (src0의 처음 16행)
   대기: DMA 완료 확인

2. 메인 루프 (행 단위 처리):
   for each 16-row 블록:
     ├── DMA 프리페치: 다음 16행을 VTCM으로 (비동기)
     │
     ├── 현재 16행 연산:
     │   for row_pair in (0, 2, 4, ..., 14):  // 2행씩 처리
     │     for col_pair in (0, 2, 4, ...):    // 2열씩 처리
     │       vec_dot_2x2(src0[row], src0[row+1], src1[col], src1[col+1])
     │       → 4개 내적 결과 동시 계산
     │     
     │     // 홀수 열 처리
     │     if (odd column):
     │       vec_dot_2x1(src0[row], src0[row+1], src1[last_col])
     │   
     │   // 홀수 행 처리
     │   if (odd row):
     │     for col: vec_dot_1x1(...)
     │
     └── DMA 완료 대기: pop으로 다음 블록 데이터 확인

3. 결과 DMA:
   dst_spad(VTCM) → DRAM (자동, hvx_vec_store_u로 직접 기록)
```

### DMA 파이프라이닝 (Double Buffering)

```c
// matmul_2d 내부 DMA 파이프라이닝 코드
static void matmul_2d(unsigned int nth, unsigned int ith, void * data) {
    dma_queue * dma = octx->ctx->dma[ith];
    
    // 1단계: 초기 프리페치 (첫 16행)
    for (uint32_t i = ir_first; i < MIN(ir_first + nrows_per_spad, ir_last); ++i) {
        src0_data_ddr = (uint8_t *) src0->data + (i * src0_row_stride);
        dma_queue_push_ddr_to_vtcm(dma, src0_spad + offset, src0_data_ddr, src0_row_size);
    }
    
    // 2단계: 연산 + 프리페치 병렬화
    for (uint32_t i = ir_first; i < ir_last; i += nrows_per_spad) {
        // 다음 블록 프리페치 (비동기)
        for (uint32_t j = 0; j < nrows_per_spad && i + nrows_per_spad + j < ir_last; ++j) {
            dma_queue_push_ddr_to_vtcm(dma, ...);  // 비동기 DMA 전송
        }
        
        // 현재 블록 연산 (DMA와 병렬 실행)
        for (uint32_t ir = 0; ir < current_block_rows; ir += 2) {
            // DMA 완료 대기 (현재 행)
            src0_row0 = dma_queue_pop(dma).dst;
            src0_row1 = dma_queue_pop(dma).dst;
            
            // 2x2 타일 연산 수행
            for (ic = 0; ic + 1 < src1_nrows; ic += 2) {
                mmctx->vec_dot_2x2(ne00, &dst[...], &dst[...],
                                   src0_row0, src0_row1,
                                   src1_spad + ic*stride, src1_spad + (ic+1)*stride);
            }
        }
    }
}
```

핵심: `dma_queue_push`는 비동기, `dma_queue_pop`은 완료 대기. 따라서 다음 블록의 DMA 전송과 현재 블록의 연산이 동시에 실행됩니다.

---

## 6.3 Flash Attention의 DMA 파이프라이닝

위치: [htp/flash-attn-ops.c](../../ggml/src/ggml-hexagon/htp/flash-attn-ops.c)

Flash Attention은 K/V를 64행(FLASH_ATTN_BLOCK_SIZE) 단위 블록으로 처리합니다:

### VTCM 레이아웃 (스레드별)

```
스레드별 VTCM 할당:
├── spad_q: Q 행 1개 (DK × sizeof(fp16))
├── spad_k: K 블록 **2개** (Double Buffer) ← 핵심!
│           = size_k_row_padded × 64 × 2
├── spad_v: V 블록 **2개** (Double Buffer) ← 핵심!
│           = size_v_row_padded × 64 × 2
├── spad_m: Mask 블록 2개 (있을 경우)
└── spad_a: 누적기 (DV × sizeof(float))
```

### 더블 버퍼링 파이프라인

```
블록 0 DMA  │ K₀→VTCM  V₀→VTCM
블록 1 DMA  │ K₁→VTCM  V₁→VTCM
            ↓
블록 0 연산 │ [Q·K₀ → softmax → ×V₀]    블록 2 DMA │ K₂→VTCM  V₂→VTCM
블록 1 연산 │ [Q·K₁ → softmax → ×V₁]    블록 3 DMA │ K₃→VTCM  V₃→VTCM
블록 2 연산 │ [Q·K₂ → softmax → ×V₂]    블록 4 DMA │ ...
```

```c
// flash-attn-ops.c 내부
// 처음 2개 블록 프리페치
for (ib = 0; ib < MIN(n_blocks, 2); ib++) {
    dma_queue_push(dma, dma_make_ptr(k_dst, k_src), ...);  // K
    dma_queue_push(dma, dma_make_ptr(v_dst, v_src), ...);  // V
    if (mask) dma_queue_push(dma, ...);                      // Mask
}

for (ib = 0; ib < n_blocks; ib++) {
    // 현재 블록 DMA 완료 대기
    k_base = dma_queue_pop(dma).dst;  // K 블록 수신
    v_base = dma_queue_pop(dma).dst;  // V 블록 수신
    m_base = mask ? dma_queue_pop(dma).dst : NULL;
    
    // 32행 단위로 QK 내적 계산 → P 확률 계산 → V 누적
    for (ic = 0; ic + VLEN_FP32 <= block_size; ic += VLEN_FP32) {
        scores = hvx_dot_f16_f16_aa_rx32(q, k + ic*stride, stride, DK, scale);
        // online softmax + V accumulate
        hvx_mad_f32_f16_aa_rx2(VKQ32, v_ptr, v_ptr+stride, p, p+1, DV);
    }
    
    // ib+2 블록 프리페치 (현재 연산과 병렬)
    if (ib + 2 < n_blocks) {
        dma_queue_push(dma, dma_make_ptr(k_base, k_src_next), ...);
        dma_queue_push(dma, dma_make_ptr(v_base, v_src_next), ...);
    }
}
```

---

## 6.4 Softmax의 VTCM 활용

위치: [htp/softmax-ops.c](../../ggml/src/ggml-hexagon/htp/softmax-ops.c)

Softmax는 DMA 파이프라이닝 없이 **스크래치패드 모드**를 사용합니다:

```
VTCM 할당 (스레드별):
├── src0_spad: 1행 크기 (작업 공간)
├── src1_spad: 1행 크기 (작업 공간)  
└── dst_spad:  1행 크기 (작업 공간)
```

각 스레드는 자신의 VTCM 영역에서 스칼라 연산을 수행합니다:
1. `hvx_fast_softmax_prep_f32`: scale + mask 적용 (입력→작업공간)
2. `hvx_fast_softmax_f32`: max-find → subtract → exp → sum → normalize (작업공간→출력)

이때는 DRAM→VTCM DMA가 아닌, L2 캐시를 통한 HVX 로드/스토어로 데이터를 직접 접근합니다. `hex_l2fetch()`를 사용하여 L2 프리페치를 수행합니다.

---

## 6.5 메모리 레이아웃과 HVX 정렬

### 정렬 요구사항

| 요소 | 정렬 | 이유 |
|------|------|------|
| HVX 벡터 로드/스토어 (aligned) | 128바이트 | `HVX_Vector *` 선언 시 자동 정렬 |
| HVX 벡터 로드/스토어 (unaligned) | 제한 없음 | `HVX_UVector *` 사용 |
| VTCM 스크래치패드 | 256바이트 | `hex_round_up(size, 256)` |
| DMA 전송 단위 | 128바이트 패딩 | `hex_round_up(row_size, 128)` |
| 메모리 할당 | 128바이트 | `ggml_backend_hexagon_buffer_type_get_alignment()` |

### 블록화 패턴

Repack 포맷의 블록 크기는 256원소(= 8 × Q4_0 블록):

```
Q4x4x2: 256 elements = 8 × 32-element blocks
         quants: 128 bytes (256 × 4bit)
         scales: 16 bytes (8 × fp16)

Q8x4x2: 256 elements = 8 × 32-element blocks  
         quants: 256 bytes (256 × 8bit) = 정확히 2 HVX 벡터
         scales: 16 bytes (8 × fp16)
```

이 크기는 HVX의 `Q6_Vw_vrmpy_VbVb` (reduce-multiply) 연산에서 8개 HVX 벡터를 한 번에 처리하는 `hvx_vec_rmpy_x8_full()` 함수와 정확히 매칭됩니다.
