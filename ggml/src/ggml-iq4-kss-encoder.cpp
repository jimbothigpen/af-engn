#include "ggml-iq4-kss-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
namespace {
struct quantize_user_data {};
#define GGML_ASSERT(x) assert(x)
static constexpr int GGML_TYPE_IQ4_KSS = 52;
static float ggml_get_quantize_fudge_factor(int) { return 1.01f; }
static size_t ggml_row_size(int, int64_t n) { return n / QK_K * sizeof(block_iq4_kss) + sizeof(float); }
static const int8_t iq4k_values[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117
};
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
static inline int popcount(uint8_t x)  { return __builtin_popcount((unsigned) x); }
static inline int popcount(uint16_t x) { return __builtin_popcount((unsigned) x); }

const uint16_t * scramble_table() {
    static std::mutex mutex;
    static std::vector<uint16_t> table;
    std::lock_guard<std::mutex> lock(mutex);
    if (table.empty()) {
        table.resize(1 << 15);
        for (int i = 0; i < int(table.size()); ++i) {
            uint16_t val = i;
            int non = popcount(val);
            if (non%2) val |= (1 << 15);
            bool found = false;
            for (int j = 0; j < int(table.size()); ++j) {
                if ((j ^ (j << 1)) == val) {
                    table[i] = j; found = true; break;
                }
            }
            if (!found) {
                printf("Oops: did not find for %d %u\n", i, val);
                exit(1);
            }
        }
    }
    return table.data();
}
uint16_t prune_iq4ks(uint16_t v, const int8_t * values, const float * x, const float * w, float dl) {
    if (popcount(v)%2 == 0) return v;
    float best_score = std::numeric_limits<float>::max();
    uint8_t q4[4];
    int jbest = -1;
    uint8_t bestq = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t q = (v >> 4*j) & 0xf;
        q4[j] = q;
        auto pc = popcount(q);
        float diff0 = dl*values[q] - x[j];
        int qmin = std::max(int(q)-2,  0);
        int qmax = std::min(int(q)+2, 15);
        for (int iq = qmin; iq <= qmax; ++iq) {
            uint8_t qq = iq;
            if (qq == q) continue;
            int pci = popcount(qq);
            if (std::abs(pci - pc)%2) {
                float diff1 = dl*values[qq] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) {
                    best_score = score; jbest = j; bestq = qq;
                }
            }
        }
    }
    GGML_ASSERT(jbest >= 0);
    q4[jbest] = bestq;
    return (q4[0] | (q4[1] << 4) | (q4[2] << 8) | (q4[3] << 12));
}
static void quantize_row_iq4_kss_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const uint16_t * table,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    *dptr = 0;
    block_iq4_kss * y = (block_iq4_kss *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    uint16_t vps[block_size/2], vms[block_size/2], vs[block_size/2];
    float xv[4], wv[4];

    float amax_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KSS);

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_kss));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
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
            float best = 0;
            float d = -max/iq4k_values[0];
            std::memset(vs, 0, block_size);
            for (int itry = -ntry; itry <= ntry; ++itry) {
                float id = (itry + values[0])/max;
                float sumqx_p = 0, sumq2_p = 0;
                float sumqx_m = 0, sumq2_m = 0;
                float this_d = 1/id;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(values,  al) << 4*j);
                        vm |= (best_index_iq4nl(values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, values, xv, wv, -this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                bool copy_p = false, copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }

                id = (itry + shifted_values[0])/max;
                this_d = 1/id;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(shifted_values,  al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, shifted_values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv, -this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = shifted_values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                copy_p = copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }
            }
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = fudge*d;
    if (!d) return;
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = (std::max(0, std::min(127, l)) << 1) - 127;
            if (l) {
                float dl = d*l;
                float idl = 1/dl;
                float mse_p = 0, mse_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = idl*xv[j];
                        vp |= (best_index_iq4nl(        values, al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, al) << 4*j);
                    }
                    vp = prune_iq4ks(vp,         values, xv, wv,  dl);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  dl);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        mse_p += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        mse_m += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                const uint16_t * v = vps;
                const int8_t * block_values = values;
                if (mse_m < mse_p) {
                    v = vms;
                    block_values = values + 16;
                }
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    for (int j = 0; j < 4; ++j) {
                        float q = block_values[(v[k] >> 4*j) & 0xf] * l;
                        sumqx += wv[j]*q*xv[j];
                        sumq2 += wv[j]*q*q;
                    }
                }
                l += 127;
                if (mse_m < mse_p) l |= 1;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    auto val = table[v[k] & 0x7fff];
                    q16[k] = (val << 1) | ((l >> k) & 1);
                }
            } else {
                l += 127;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    q16[k] = ((l >> k) & 1);
                }
            }
        }
    }
    if (sumq2 > 0) *dptr = fudge*sumqx/sumq2;
}
}

extern "C" size_t ggml_quantize_iq4_kss_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(nrows >= 0 && nrows <= INT_MAX);
    assert(n_per_row > 0 && n_per_row <= INT_MAX && n_per_row % 256 == 0);
    assert(src && dst);
    constexpr int kBlockSize = 32;
    std::vector<float> all_scales(n_per_row/kBlockSize);
    float weight[kBlockSize];
    const uint16_t * table = scramble_table();
    QHelper helper(imatrix, nullptr, (int) n_per_row, kBlockSize);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);
    auto q_func = [&all_scales, &weight, table] (const float * x, void * vy, int n, const float * weights, const quantize_user_data *) {
        quantize_row_iq4_kss_impl(n, x, (char *) vy, all_scales.data(), weight, iq4k_values, weights, table, 7);
    };
    helper.quantize((int) nrows, src, dst, (int) row_size, q_func);
    return nrows * row_size;
}
