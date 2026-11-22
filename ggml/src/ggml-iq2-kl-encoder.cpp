// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// IQ2_KL scalar encoder, ported from ik_llama.cpp @ iqk_quantize.cpp
// (quantize_row_iq2_kl_impl, quantize_row_iq2_kl_ref, ggml_quantize_iq2_kl_scalar).
// MIT-licensed; attribution preserved.

#include "ggml-iq2-kl-encoder.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

using ggml_fp16_t = uint16_t;
struct quantize_user_data {};
#define GGML_ASSERT(x) assert(x)

static constexpr int GGML_TYPE_IQ2_KL = 58;
static float ggml_get_quantize_fudge_factor(int) { return 1.025f; }
static size_t ggml_row_size(int, int64_t n) { return sizeof(ggml_fp16_t) + (n / QK_K) * sizeof(block_iq2_kl); }

static const int8_t iq3nl_values[16] = {
    -63, -40, -23, -10, 1, 13, 28,  47,
    -59, -36, -19,  -6, 5, 17, 32,  51,
};

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

const int8_t iq3nl_index[111] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
    9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
    4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};

inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}

static const int k_index[64] = {
    -1, -2, 0, -3, -4, 1, -5, -6, 2, -7, -8, 3, -9, 4, -10, 5,
    -11, 6, 7, -12, 8, 9, 10, -13, 11, -14, -15, -16, 12, 13, -17, 14,
    -18, -19, 15, 16, 17, 18, 19, -20, -21, 20, 21, 22, 23, 24, -22, -23,
    25, -24, 26, -25, 27, -26, 28, 29, -27, -28, 30, -29, -30, 31, -31, -32
};

static const std::vector<std::vector<int>> k_neighbours = {
    { 2, 0, 6, 11, 7, 3, 8, 15,  },
    { 0, 2, 3, 6, 7, 1, 8, 4,  },
    { 0, 1, 3, 4, 8, 7, 9, 6,  },
    { 1, 0, 3, 4, 8, 9, 7, 10,  },
    { 1, 4, 5, 10, 9, 3, 8, 0,  },
    { 5, 1, 4, 10, 9, 14, 8, 3,  },
    { 6, 2, 7, 0, 3, 11, 8, 15,  },
    { 3, 7, 0, 6, 8, 4, 12, 9,  },
    { 3, 4, 8, 9, 1, 7, 12, 10,  },
    { 4, 10, 5, 9, 1, 8, 13, 14,  },
    { 11, 2, 6, 7, 20, 15, 25, 21,  },
    { 8, 7, 3, 12, 9, 16, 17, 13,  },
    { 14, 5, 10, 19, 9, 13, 4, 18,  },
    { 6, 15, 7, 11, 20, 21, 16, 2,  },
    { 15, 7, 16, 6, 21, 12, 17, 22,  },
    { 12, 16, 17, 8, 15, 7, 13, 22,  },
    { 19, 10, 13, 18, 14, 9, 12, 24,  },
    { 11, 20, 25, 6, 15, 2, 21, 7,  },
    { 20, 15, 21, 6, 11, 7, 16, 26,  },
    { 14, 19, 29, 10, 28, 18, 13, 24,  },
    { 25, 11, 20, 21, 15, 6, 26, 30,  },
    { 19, 24, 28, 18, 29, 23, 13, 17,  },
    { 29, 19, 14, 28, 24, 18, 10, 13,  },
    { 20, 26, 21, 25, 30, 15, 22, 16,  },
    { 27, 26, 22, 23, 21, 30, 16, 24,  },
    { 27, 24, 28, 31, 23, 18, 22, 17,  },
    { 25, 30, 20, 26, 21, 11, 15, 22,  },
    { 30, 26, 25, 20, 21, 27, 22, 15,  },
    { 30, 27, 31, 26, 22, 23, 21, 24,  },
    { 31, 27, 30, 26, 28, 23, 22, 24,  },
    { 31, 28, 29, 27, 24, 23, 19, 18,  },
    { 29, 28, 31, 24, 19, 27, 14, 18,  },
};

void quantize_row_iq2_kl_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales) {
    constexpr int kBlockSize = 32;
    constexpr float kSigmaFactor = 2.25f;
    constexpr int ntry = 5;

    auto values = iq3nl_values;
    std::pair<int8_t, int8_t> grid[32];
    for (int j = 0; j < 64; ++j) {
        if (int i = k_index[j]; i >= 0) {
            int i1 = j/8, i2 = j%8;
            grid[i] = {values[i1], values[i2]};
        }
    }

    ggml_half * dptr = (ggml_half *)vy;
    auto y = (block_iq2_kl *)(dptr + 1);

    float weight[kBlockSize];

    auto index = [&grid, values] (float id, float x1, float x2, float w1, float w2) {
        float sx1 = id*x1;
        float sx2 = id*x2;
        int l1 = best_index_iq3nl(values, sx1);
        int l2 = best_index_iq3nl(values, sx2);
        int i = k_index[8*l1 + l2];
        if (i >= 0) return i;
        auto& neigh = k_neighbours[-i-1];
        float best = std::numeric_limits<float>::max();
        int ibest = -1;
        for (auto& n : neigh) {
            float diff1 = grid[n].first  - sx1;
            float diff2 = grid[n].second - sx2;
            float score = w1*diff1*diff1 + w2*diff2*diff2;
            if (score < best) {
                best = score; ibest = n;
            }
        }
        GGML_ASSERT(ibest >= 0);
        return ibest;
    };

    float max_scale = 0, max_abs_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_KL);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq2_kl));
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]);
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < kBlockSize; j += 2) {
                float w1 = weight[j+0];
                float w2 = weight[j+1];
                int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                float q1 = grid[idx].first ;
                float q2 = grid[idx].second;
                sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_p += w1*q1*q1 + w2*q2*q2;
                idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                q1 = grid[idx].first ;
                q2 = grid[idx].second;
                sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_m += w1*q1*q1 + w2*q2*q2;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; j += 2) {
                    float w1 = weight[j+0];
                    float w2 = weight[j+1];
                    int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                    float q1 = grid[idx].first ;
                    float q2 = grid[idx].second;
                    sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_p += w1*q1*q1 + w2*q2*q2;
                    idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                    q1 = grid[idx].first ;
                    q2 = grid[idx].second;
                    sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_m += w1*q1*q1 + w2*q2*q2;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m;
                }
            }
            scales[ib] = d;
            float ad = std::abs(d);
            if (ad > max_abs_scale) {
                max_abs_scale = ad; max_scale = d;
            }
        }
    }

    if (!max_abs_scale) {
        dptr[0] = GGML_FP32_TO_FP16(0.f);
        return;
    }

    float d = -max_scale/32;
    float id = 1/d;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]);
            }
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-32, std::min(31, ls));
            int lsmin = std::max(-32, ls-1);
            int lsmax = std::min( 31, ls+1);
            float best_score = std::numeric_limits<float>::max();
            int best_ls = ls;
            for (int ils = lsmin; ils <= lsmax; ++ils) {
                float dl = d*ils;
                float idl = dl ? 1/dl : 0.f;
                float score = 0;
                for (int j = 0; j < kBlockSize/2; ++j) {
                    float w1 = weight[2*j+0];
                    float w2 = weight[2*j+1];
                    int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                    float diff1 = dl*grid[idx].first  - xb[2*j+0];
                    float diff2 = dl*grid[idx].second - xb[2*j+1];
                    score += w1*diff1*diff1 + w2*diff2*diff2;
                }
                if (score < best_score) {
                    best_score = score;
                    best_ls = ils;
                }
            }
            ls = best_ls;
            int uls = ls + 32;
            y[ibl].scales_l[ib%4] |= ((uls & 0xf) << 4*(ib/4));
            y[ibl].scales_h |= ((uls >> 4) << 2*ib);
            if (ls == 0) continue;
            float dl = d*ls;
            float idl = 1/dl;
            for (int j = 0; j < kBlockSize/2; ++j) {
                float w1 = weight[2*j+0];
                float w2 = weight[2*j+1];
                int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                y[ibl].qs[16*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float q1 = ls*grid[idx].first ;
                float q2 = ls*grid[idx].second;
                sumqx += w1*q1*xb[2*j] + w2*q2*xb[2*j+1];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) d = sumqx/sumq2;

    dptr[0] = GGML_FP32_TO_FP16(fudge * d);
}

size_t quantize_iq2_kl(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
                       const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row % QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KL, n_per_row);
    int nblock = n_per_row / QK_K;
    std::vector<float> all_scales(nblock * (QK_K / kBlockSize));
    auto q_func = [&all_scales] (const float * x, void * vy, int n_per_row, const float * imatrix,
                                 [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq2_kl_impl(x, vy, n_per_row, imatrix, all_scales.data());
    };
    QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

} // namespace

extern "C" size_t ggml_quantize_iq2_kl_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(nrows >= 0 && nrows <= INT_MAX);
    assert(n_per_row > 0 && n_per_row <= INT_MAX && n_per_row % 256 == 0);
    assert(src && dst);
    return quantize_iq2_kl(src, dst, nrows, n_per_row, imatrix, nullptr);
}
