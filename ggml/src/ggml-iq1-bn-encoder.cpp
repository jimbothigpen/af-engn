#include "ggml-iq1-bn-encoder.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace {

void pack_1bn(const int8_t * q, block_iq1_bn & y) {
    static const int k_nb[6] = {1, 3, 9, 27, 81, 243};
    std::memset(&y, 0, sizeof(block_iq1_bn));
    int v13 = 0;
    for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
        for (int k = 0; k < 3; ++k) {
            int idx = 0;
            for (int j = 0; j < 5; ++j) {
                idx += k_nb[j]*q[16*i16 + 5*k + j];
            }
            y.ql[3*i16 + k] = (256*idx + k_nb[5] - 1)/k_nb[5];
        }
        v13 += k_nb[i16]*q[16*i16 + 15];
    }
    y.extra = (256*v13 + k_nb[5] - 1)/k_nb[5];
}

void quantize_one_row(const float * src, void * vrow, int64_t n_per_row) {
    ggml_half * dptr = (ggml_half *)vrow;
    block_iq1_bn * y = (block_iq1_bn *)(dptr + 1);

    float max = 0;
    for (int64_t j = 0; j < n_per_row; ++j) max = std::max(max, fabsf(src[j]));
    ggml_half d = GGML_FP32_TO_FP16(max);
    std::memcpy(dptr, &d, sizeof(d));

    const float thresh = 0.5f*max;
    const int64_t nblock = n_per_row/QK_IQ1BN;
    int8_t L[QK_IQ1BN];
    for (int64_t ib = 0; ib < nblock; ++ib) {
        const float * xb = src + ib*QK_IQ1BN;
        for (int j = 0; j < QK_IQ1BN; ++j) {
            const float v = xb[j];
            L[j] = fabsf(v) < thresh ? 1 : v < 0 ? 0 : 2;
        }
        pack_1bn(L, y[ib]);
    }
}

}

size_t ggml_quantize_iq1_bn_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_IQ1BN == 0);
    const size_t row_size = 2 + sizeof(block_iq1_bn)*(size_t)(n_per_row/QK_IQ1BN);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_one_row(src + row*n_per_row, qrow, n_per_row);
        qrow += row_size;
    }
    return (size_t)nrows*row_size;
}
