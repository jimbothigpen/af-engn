// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// IQ2_K scalar encoder, ported from ik_llama.cpp @ iqk_quantize.cpp
// (quantize_row_iq2_k_impl, quantize_row_iq2_k_ref, ggml_quantize_iq2_k_scalar).
// MIT-licensed; attribution preserved.
#include "ggml-iq2-k-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <array>
#include <utility>
#include <climits>
namespace {
using ggml_fp16_t = uint16_t;
struct quantize_user_data {};
#define GGML_ASSERT(x) assert(x)
static constexpr int GGML_TYPE_IQ2_K = 45;
static float ggml_get_quantize_fudge_factor(int) { return 1.0f; }
static size_t ggml_row_size(int, int64_t n) { return n / QK_K * sizeof(block_iq2_k); }
static const int8_t iq2nl_values[8] = { -31, -13, 1, 17, -26, -8, 6, 22 };
static inline float fp32_from_bits(uint32_t w) {
    union { uint32_t as_bits; float as_value; } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}
static inline uint32_t fp32_to_bits(float fv) {
    union { float as_value; uint32_t as_bits; } fp32;
    fp32.as_value = fv;
    return fp32.as_bits;
}
static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float fv) {
    const float scale_to_inf  = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(fv) * scale_to_inf) * scale_to_zero;
    const uint32_t w = fp32_to_bits(fv);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }
    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
float make_qx_quants(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax; max = x[i];
        }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax-1, l));
        L[i] = l + nmax;
        sumlx += qw[i]*x[i]*l;
        suml2 += qw[i]*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax-1, l));
            sumlx += qw[i]*x[i]*l;
            suml2 += qw[i]*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

struct QHelper {
    QHelper(const float * imatrix, const quantize_user_data * user_data, int n_per_row, int block_size) :
        m_imatrix(imatrix), m_user_data(user_data), m_n_per_row(n_per_row), m_block_size(block_size) {
        if (m_imatrix) {
            m_weight.resize(m_n_per_row);
        }
    }
    const float * row_weights(const float * x) {
        constexpr float kEps  = 1e-9f;
        constexpr float kEps2 = kEps*kEps;
        if (!m_imatrix) return m_imatrix;
        int nblock = m_n_per_row / m_block_size;
        for (int ib = 0; ib < nblock; ++ib) {
            auto wb_in = m_imatrix + ib*m_block_size;
            auto xb = x + ib*m_block_size;
            auto wb = m_weight.data() + ib*m_block_size;
            float sumw2 = 0, sumx2 = 0, sumwx = 0;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = wb_in[j];
                sumw2 += wb[j]*wb[j];
                sumx2 += xb[j]*xb[j];
                sumwx += wb[j]*std::abs(xb[j]);
            }
            if (sumw2 > m_block_size*kEps2 && sumx2 > m_block_size*kEps2 && sumwx > m_block_size*kEps2) continue;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = kEps;
            }
        }
        return m_weight.data();
    }
    template <typename Func>
    void quantize(int nrows, const float * src, void * dst, int row_size, const Func& qfunc) {
        auto cdst = (char *)dst;
        for (int row = 0; row < nrows; ++row) {
            auto weights = row_weights(src);
            qfunc(src, cdst, m_n_per_row, weights, m_user_data);
            src  += m_n_per_row;
            cdst += row_size;
        }
    }
private:
    const float * m_imatrix;
    const quantize_user_data * m_user_data;
    const int     m_n_per_row;
    const int     m_block_size;
    std::vector<float> m_weight;
};
namespace {

inline int best_index_iq2nl(const int8_t * values, float x) {
    int idx = x < values[1] ? 0 : x > values[2] ? 2 : 1;
    return x - values[idx] < values[idx+1] - x ? idx : idx + 1;
}

void quantize_row_iq2_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {

    constexpr int kBlockSize = 16;

    block_iq2_k * y = (block_iq2_k *)vy;

    float scales[QK_K/kBlockSize];
    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];
    float sw[QK_K/kBlockSize];
    int8_t Ls[QK_K/kBlockSize];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    const int8_t * shifted_values = iq2nl_values + 4;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_K);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            sw[ib] = 0;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j+1] = sumw[j] + weight[jj];
                sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kBlockSize; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = i2; i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[0] + (sumx[i2] - sumx[i1])*iq2nl_values[1]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[2] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[0]*iq2nl_values[0] + (sumw[i2] - sumw[i1])*iq2nl_values[1]*iq2nl_values[1]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[2]*iq2nl_values[2] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[3]*iq2nl_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[0] + (sumx[i2] - sumx[i1])*shifted_values[1]
                              + (sumx[i3] - sumx[i2])*shifted_values[2] + (sumx[kBlockSize] - sumx[i3])*shifted_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[0]*shifted_values[0] + (sumw[i2] - sumw[i1])*shifted_values[1]*shifted_values[1]
                              + (sumw[i3] - sumw[i2])*shifted_values[2]*shifted_values[2] + (sumw[kBlockSize] - sumw[i3])*shifted_values[3]*shifted_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[3] + (sumx[i2] - sumx[i1])*iq2nl_values[2]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[1] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[3]*iq2nl_values[3] + (sumw[i2] - sumw[i1])*iq2nl_values[2]*iq2nl_values[2]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[1]*iq2nl_values[1] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[0]*iq2nl_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[3] + (sumx[i2] - sumx[i1])*shifted_values[2]
                              + (sumx[i3] - sumx[i2])*shifted_values[1] + (sumx[kBlockSize] - sumx[i3])*shifted_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[3]*shifted_values[3] + (sumw[i2] - sumw[i1])*shifted_values[2]*shifted_values[2]
                              + (sumw[i3] - sumw[i2])*shifted_values[1]*shifted_values[1] + (sumw[kBlockSize] - sumw[i3])*shifted_values[0]*shifted_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }

        if (!max_abs_scale) continue;
        float d = make_qx_quants(QK_K/kBlockSize, 8, scales, Ls, sw);
        if (!d) continue;

        //float d = -max_scale/8;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-8, std::min(7, ls));
            y[ibl].scales[ib/2] |= ((ls + 8) << 4*(ib%2));
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib32%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(fudge*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}
}

size_t quantize_iq2_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data);

void quantize_row_iq2_k_ref(const float * x, block_iq2_k  * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_k(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_iq2_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_k * y = (block_iq2_k *)vy;
    quantize_row_iq2_k_ref(x, y, k);
}

size_t quantize_iq2_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    QHelper helper(imatrix, user_data, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq2_k_impl);
    return nrows * row_size;
}

}

extern "C" size_t ggml_quantize_iq2_k_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(nrows >= 0 && nrows <= INT_MAX);
    assert(n_per_row > 0 && n_per_row <= INT_MAX && n_per_row % 256 == 0);
    assert(src && dst);
    return quantize_iq2_k(src, dst, nrows, n_per_row, imatrix, nullptr);
}
