#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t ggml_quantize_pq2_0_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void quantize_row_pq2_0_ref(const float * x, void * y, int64_t k);

#ifdef __cplusplus
}
#endif
