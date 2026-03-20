#pragma once

// Queue-based DMA for DDR <-> VTCM transfers
// Adapted from llama.cpp hex-dma.h/c (ggml/src/ggml-hexagon/htp/hex-dma.h)
// Uses dmstart/dmlink chaining for pipelined async DMA.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Descriptor layout  (matches Hexagon UDMA type-1 / 2D descriptor)
// ---------------------------------------------------------------------------
typedef struct dma_queue_desc {
  void    *next;
  uint32_t length;       // unused for type-1
  uint32_t desctype : 2; // 1 = type-1 (2D)
  uint32_t dstcoherent : 1;
  uint32_t srccoherent : 1;
  uint32_t dstbypass : 1;
  uint32_t srcbypass : 1;
  uint32_t order     : 1;
  uint32_t dstate    : 1;
  uint32_t _pad0     : 24;
  void    *src;
  void    *dst;
  uint32_t allocation : 2;
  uint32_t padding    : 30;
  uint16_t roiwidth;
  uint16_t roiheight;
  uint16_t srcstride;
  uint16_t dststride;
  uint16_t srcwidthoffset;
  uint16_t dstwidthoffset;
} __attribute__((aligned(64))) dma_queue_desc_t;

#define DMA_QUEUE_DESC_DSTATE_INCOMPLETE 0
#define DMA_QUEUE_DESC_DSTATE_COMPLETE   1
#define DMA_QUEUE_DESC_DESCTYPE_TYPE1    1

// ---------------------------------------------------------------------------
// Pointer pair (returned by pop to identify completed transfer)
// ---------------------------------------------------------------------------
typedef struct {
  void       *dst;
  const void *src;
} dma_ptr_t;

static inline dma_ptr_t dma_make_ptr(void *dst, const void *src) {
  dma_ptr_t p = { dst, src };
  return p;
}

// ---------------------------------------------------------------------------
// Queue structure
// ---------------------------------------------------------------------------
typedef struct dma_queue {
  dma_queue_desc_t *desc;     // descriptor ring
  dma_queue_desc_t *tail;     // last linked descriptor
  dma_ptr_t        *dptr;     // dst/src bookkeeping
  uint32_t          push_idx;
  uint32_t          pop_idx;
  uint32_t          capacity;
  uint32_t          idx_mask;
} dma_queue_t;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
dma_queue_t *dma_queue_create(uint32_t capacity);
void         dma_queue_delete(dma_queue_t *q);
void         dma_queue_flush(dma_queue_t *q);

// ---------------------------------------------------------------------------
// Low-level DMA asm wrappers
// ---------------------------------------------------------------------------
static inline void dma_q_dmstart(void *next) {
  asm volatile("release(%0):at" ::"r"(next));
  asm volatile("dmstart(%0)"   ::"r"(next));
}

static inline void dma_q_dmlink(void *cur, void *next) {
  asm volatile("release(%0):at" ::"r"(next));
  asm volatile("dmlink(%0, %1)" ::"r"(cur), "r"(next));
}

static inline unsigned int dma_q_dmpoll(void) {
  unsigned int ret = 0;
  asm volatile("%0 = dmpoll" : "=r"(ret) :: "memory");
  return ret;
}

static inline unsigned int dma_q_dmwait(void) {
  unsigned int ret = 0;
  asm volatile("%0 = dmwait" : "=r"(ret) :: "memory");
  return ret;
}

// ---------------------------------------------------------------------------
// Push – enqueue a 2D DMA transfer (1D is just height=1)
// ---------------------------------------------------------------------------
static inline bool dma_queue_push(dma_queue_t *q,
                                  dma_ptr_t    dptr,
                                  size_t       dst_stride,
                                  size_t       src_stride,
                                  size_t       width,   // bytes per row
                                  size_t       nrows) {
  if (((q->push_idx + 1) & q->idx_mask) == q->pop_idx) {
    return false;  // queue full
  }

  dma_queue_desc_t *desc = &q->desc[q->push_idx];

  desc->next           = 0;
  desc->length         = 0;
  desc->desctype       = DMA_QUEUE_DESC_DESCTYPE_TYPE1;
  desc->dstbypass      = 1;
  desc->srcbypass      = 1;
  desc->dstcoherent    = 0;
  desc->srccoherent    = 0;
  desc->order          = 0;
  desc->dstate         = DMA_QUEUE_DESC_DSTATE_INCOMPLETE;
  desc->src            = (void *) dptr.src;
  desc->dst            = (void *) dptr.dst;
  desc->allocation     = 0;
  desc->padding        = 0;
  desc->roiwidth       = (uint16_t) width;
  desc->roiheight      = (uint16_t) nrows;
  desc->srcstride      = (uint16_t) src_stride;
  desc->dststride      = (uint16_t) dst_stride;
  desc->srcwidthoffset = 0;
  desc->dstwidthoffset = 0;

  q->dptr[q->push_idx] = dptr;

  dma_q_dmlink(q->tail, desc);
  q->tail = desc;

  q->push_idx = (q->push_idx + 1) & q->idx_mask;
  return true;
}

// ---------------------------------------------------------------------------
// Convenience: 1D DDR -> VTCM
// ---------------------------------------------------------------------------
static inline bool dma_queue_push_ddr_to_vtcm_1d(dma_queue_t *q,
                                                  dma_ptr_t    dptr,
                                                  size_t       size) {
  return dma_queue_push(q, dptr, size, size, size, 1);
}

// ---------------------------------------------------------------------------
// Convenience: 2D DDR -> VTCM
// ---------------------------------------------------------------------------
static inline bool dma_queue_push_ddr_to_vtcm_2d(dma_queue_t *q,
                                                  dma_ptr_t    dptr,
                                                  size_t       width,
                                                  size_t       height,
                                                  size_t       src_stride,
                                                  size_t       dst_stride) {
  return dma_queue_push(q, dptr, dst_stride, src_stride, width, height);
}

// ---------------------------------------------------------------------------
// Pop – wait for the oldest enqueued transfer to complete
// ---------------------------------------------------------------------------
static inline dma_ptr_t dma_queue_pop(dma_queue_t *q) {
  dma_ptr_t dptr = { 0, 0 };

  if (q->push_idx == q->pop_idx) {
    return dptr;  // empty
  }

  dma_queue_desc_t *desc = &q->desc[q->pop_idx];

  // Spin until descriptor is complete
  while (1) {
    dma_q_dmpoll();
    if (desc->dstate == DMA_QUEUE_DESC_DSTATE_COMPLETE) {
      break;
    }
  }

  dptr = q->dptr[q->pop_idx];
  q->pop_idx = (q->pop_idx + 1) & q->idx_mask;
  return dptr;
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------
static inline bool dma_queue_empty(dma_queue_t *q) {
  return q->push_idx == q->pop_idx;
}

static inline uint32_t dma_queue_depth(dma_queue_t *q) {
  return (q->push_idx - q->pop_idx) & q->idx_mask;
}

#ifdef __cplusplus
}
#endif
