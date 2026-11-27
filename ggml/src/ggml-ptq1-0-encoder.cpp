#include "ggml-ptq1-0-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static const size_t ptq1_0_stages[2] = {16, 8};

void quantize_row_ptq1_0_ref(const float * x, void * vy, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;
    block_ptq1_0 * y = (block_ptq1_0 *)vy;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PTQ1_0; j++) {
            const float a = fabsf(x[j]);
            amax = amax > a ? amax : a;
        }

        const float d  = amax;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        size_t j = 0;
        for (size_t s = 0; s < 2; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(y->qs); j += c) {
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = 0;
                    for (size_t n = 0; n < 5; ++n) {
                        int xi = (int)lroundf(x[m + n*c] * id) + 1;
                        q *= 3;
                        q += (uint8_t)xi;
                    }
                    q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
                    y[i].qs[j + m] = q;
                }
                x += 5 * c;
            }
        }
        for (size_t h = 0; h < sizeof(y->qh); ++h) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                int xi = (int)lroundf(x[h + m*sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += (uint8_t)xi;
            }
            q *= 3;
            q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
            y[i].qh[h] = q;
        }
        x += 4 * sizeof(y->qh);
    }
}

size_t ggml_quantize_ptq1_0_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    const size_t row_size = (size_t)(n_per_row / QK_PTQ1_0) * sizeof(block_ptq1_0);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_ptq1_0_ref(src, qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return (size_t)nrows * row_size;
}
