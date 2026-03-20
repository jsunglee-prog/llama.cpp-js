#include "dsp/dma_queue.h"

#include <HAP_farf.h>
#include <stdlib.h>
#include <string.h>

// Round up to next power of two
static inline uint32_t pow2_ceil(uint32_t x) {
  if (x <= 1) return 1;
  int p = 2;
  x--;
  while (x >>= 1) { p <<= 1; }
  return p;
}

dma_queue_t *dma_queue_create(uint32_t capacity) {
  dma_queue_t *q = (dma_queue_t *) memalign(32, sizeof(dma_queue_t));
  if (!q) {
    FARF(ERROR, "%s: failed to allocate DMA queue", __FUNCTION__);
    return NULL;
  }

  capacity = pow2_ceil(capacity);

  memset(q, 0, sizeof(dma_queue_t));
  q->capacity = capacity;
  q->idx_mask = capacity - 1;

  q->desc = (dma_queue_desc_t *) memalign(64, capacity * sizeof(dma_queue_desc_t));
  if (!q->desc) {
    FARF(ERROR, "%s: failed to allocate descriptors", __FUNCTION__);
    free(q);
    return NULL;
  }
  memset(q->desc, 0, capacity * sizeof(dma_queue_desc_t));

  q->dptr = (dma_ptr_t *) memalign(4, capacity * sizeof(dma_ptr_t));
  if (!q->dptr) {
    FARF(ERROR, "%s: failed to allocate dma ptr array", __FUNCTION__);
    free(q->desc);
    free(q);
    return NULL;
  }
  memset(q->dptr, 0, capacity * sizeof(dma_ptr_t));

  // tail points to last slot (sentinel for first dmlink)
  q->tail = &q->desc[capacity - 1];

  FARF(HIGH, "dma_queue_create: capacity %u", capacity);
  return q;
}

void dma_queue_delete(dma_queue_t *q) {
  if (!q) return;
  free(q->desc);
  free(q->dptr);
  free(q);
}

void dma_queue_flush(dma_queue_t *q) {
  while (dma_queue_pop(q).dst != NULL)
    ;
}
