#include "ggml-iq4-k-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <climits>
namespace {
using ggml_fp16_t = uint16_t;
struct quantize_user_data {};
#define GGML_ASSERT(x) assert(x)
#define MAX(a,b) std::max((a),(b))
#define MIN(a,b) std::min((a),(b))
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
static constexpr int GGML_TYPE_IQ4_K = 44;
static float ggml_get_quantize_fudge_factor(int) { return 1.0f; }
static size_t ggml_row_size(int, int64_t n) { return n / 256 * 144; }
static const int8_t iq4k_values[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117
};
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
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

const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}
namespace {
static void quantize_row_iq4_k_impl_bs16(const float * x,
        block_iq4_k * y,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 16;

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(y, 0, sizeof(block_iq4_k));
    y->d = GGML_FP32_TO_FP16(0.f);

    uint16_t * scales_h = (uint16_t *)y->scales_h;

    const int8_t * shifted_values = values + 16;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_K);

#if 0
    float f_values[32];
    for (int j = 0; j < 32; ++j) f_values[j] = values[j];
#endif

    float max_scale = 0, amax_scale = 0;
    uint16_t extra = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
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
        bool is_shifted = false;
#if 0
        float best = 0;
        __m256 vx[block_size/8], vw[block_size/8];
        for (int j = 0; j < block_size/8; ++j) vx[j] = _mm256_loadu_ps(xb + 8*j);
        for (int j = 0; j < block_size/8; ++j) vw[j] = _mm256_loadu_ps(weight + 8*j);
        (void)evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best);
#else
        float sumqx_p = 0, sumq2_p = 0;
        float sumqx_m = 0, sumq2_m = 0;
        for (int j = 0; j < block_size; ++j) {
            float w = weight[j];
            float al = id*xb[j];
            int l = best_index_iq4nl(values, al);
            float q = values[l];
            sumqx_p += w*q*xb[j];
            sumq2_p += w*q*q;
            l = best_index_iq4nl(values, -al);
            q = values[l];
            sumqx_m += w*q*xb[j];
            sumq2_m += w*q*q;
        }
        d = sumqx_p/sumq2_p;
        float best = d*sumqx_p;
        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            d = sumqx_m/sumq2_m; best = d*sumqx_m;
        }
#endif
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
#if 0
            if (evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
                is_shifted = false;
            }
#else
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
            }
#endif
            id = (itry + shifted_values[0])/max;
#if 0
            if (evaluate_block_iq4k<block_size>(id, values+16, f_values+16, vx, vw, d, best)) {
                is_shifted = true;
            }
#else
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(shifted_values, al);
                float q = shifted_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(shifted_values, -al);
                q = shifted_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
            }
#endif
        }
        if (is_shifted) extra |= (1 << ib);
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }
    float d = -max_scale/32;
    y->d = GGML_FP32_TO_FP16(fudge*d);
    y->extra = extra;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const int8_t * block_values = extra & (1 << ib) ? shifted_values : values;
        int l = nearest_int(id*scales[ib]);
        l = MAX(-32, MIN(31, l));
        float dl = d * l;
        float idl = dl ? 1/dl : 0.f;
        uint8_t * Lb = L + ib*block_size;
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        for (int j = 0; j < block_size; ++j) {
            Lb[j] = best_index_iq4nl(block_values, idl*xb[j]);
            float w = weight[j];
            float q = block_values[Lb[j]]*l;
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        l += 32;
        uint8_t l_l = l & 0xf;
        uint8_t l_h = l >>  4;
        if (ib%2 == 0) y->scales_l[ib/2] = l_l;
        else y->scales_l[ib/2] |= (l_l << 4);
        scales_h[ib/8] |= (l_h << 2*(ib%8));
    }
    if (sumq2 > 0) y->d = GGML_FP32_TO_FP16(fudge*sumqx/sumq2);

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y->qs[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

}

size_t quantize_iq4_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    uint8_t L[QK_K];
    float weight[16];
    float scales[QK_K/16];
    auto q_func = [&L, &weight, &scales] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        block_iq4_k * iq4 = (block_iq4_k *)vy;
        int nblock = n_per_row/QK_K;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = imatrix ? imatrix + QK_K*ibl : nullptr;
            quantize_row_iq4_k_impl_bs16(x + QK_K*ibl, iq4 + ibl,
                    scales, weight, L, iq4k_values, qw, 7);
        }
    };
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_K, n_per_row);
    QHelper helper(imatrix, user_data, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}


}
extern "C" size_t ggml_quantize_iq4_k_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(nrows >= 0 && nrows <= INT_MAX);
    assert(n_per_row > 0 && n_per_row <= INT_MAX && n_per_row % 256 == 0);
    assert(src && dst);
    return quantize_iq4_k(src, dst, nrows, n_per_row, imatrix, nullptr);
}
