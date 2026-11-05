#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t ggml_quantize_iq4_k_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

static inline void quantize_row_iq4_k_ref(const float * src, void * dst, int64_t n) {
    ggml_quantize_iq4_k_scalar(src, dst, 1, n, NULL);
}

#ifdef __cplusplus
}
#endif
