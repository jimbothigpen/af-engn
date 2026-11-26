#include "ggml-iq2-bn-encoder.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace {

void quantize_one_row(const float * src, void * vrow, int64_t n_per_row) {
    constexpr int Nj = QK_IQ2BN / 4;

    float max = 0.0f;
    for (int64_t j = 0; j < n_per_row; ++j) {
        max = std::max(max, fabsf(src[j]));
    }
    std::memcpy(vrow, &max, sizeof(max));
    block_iq2_bn * y = (block_iq2_bn *)((char *)vrow + sizeof(max));

    const float thresh = 0.5f * max;
    const int64_t nblock = n_per_row / QK_IQ2BN;
    uint8_t L[QK_IQ2BN];
    for (int64_t ib = 0; ib < nblock; ++ib) {
        const float * xb = src + ib * QK_IQ2BN;
        for (int j = 0; j < QK_IQ2BN; ++j) {
            const float v = xb[j];
            L[j] = fabsf(v) < thresh ? 1 : (v < 0 ? 0 : 2);
        }
        for (int j = 0; j < Nj; ++j) {
            y[ib].qs[j] = (uint8_t)(L[j] | (L[j + Nj] << 2) | (L[j + 2 * Nj] << 4) | (L[j + 3 * Nj] << 6));
        }
    }
}

}

size_t ggml_quantize_iq2_bn_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_IQ2BN == 0);
    const size_t row_size = sizeof(float) + sizeof(block_iq2_bn) * (size_t)(n_per_row / QK_IQ2BN);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_one_row(src + row * n_per_row, qrow, n_per_row);
        qrow += row_size;
    }
    return (size_t)nrows * row_size;
}
