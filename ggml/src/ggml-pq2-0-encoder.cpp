#include "ggml-pq2-0-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <string.h>

void quantize_row_pq2_0_ref(const float * x, void * vy, int64_t k) {
    static const int qk = QK_PQ2_0;
    assert(k % qk == 0);

    block_pq2_0 * y = (block_pq2_0 *)vy;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float a = fabsf(x[i * qk + j]);
            if (a > amax) amax = a;
        }
        const float d  = amax;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        memset(y[i].qs, 0, sizeof(y[i].qs));

        for (int j = 0; j < qk; ++j) {
            const float w = x[i * qk + j];
            int q = (int)roundf(w * id) + 1;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            y[i].qs[j / 4] |= (uint8_t)(q << ((j % 4) * 2));
        }
    }
}

size_t ggml_quantize_pq2_0_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    const size_t row_size = (size_t)(n_per_row / QK_PQ2_0) * sizeof(block_pq2_0);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_pq2_0_ref(src, qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return (size_t)nrows * row_size;
}
