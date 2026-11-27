#include "ggml-q6-0-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"

#include <assert.h>
#include <cmath>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define GROUP_MAX_EPS_Q6_0 1e-15f

static inline int q6_0_nearest_int_mul(float a, float b) {
    float val = std::fma(a, b, 12582912.f);
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float q6_0_make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
        const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS_Q6_0) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = q6_0_nearest_int_mul(iscale, x[i]);
            L[i] = (int8_t)(nmax + MAX(-nmax, MIN(nmax - 1, l)));
        }
        return 1 / iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = q6_0_nearest_int_mul(iscale, x[i]);
        l = MAX(-nmax, MIN(nmax - 1, l));
        L[i] = (int8_t)(l + nmax);
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx = std::fma(w * x[i], (float)l, sumlx);
        suml2 = std::fma(w * (float)l, (float)l, suml2);
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
    float best = scale * sumlx;
    float best_sumlx = sumlx, best_suml2 = suml2;
    for (int is = -9; is <= 9; ++is) {
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = q6_0_nearest_int_mul(iscale, x[i]);
            l = MAX(-nmax, MIN(nmax - 1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += (w * x[i]) * (float)l;
            suml2 += (w * (float)l) * (float)l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = q6_0_nearest_int_mul(iscale, x[i]);
                L[i] = (int8_t)(nmax + MAX(-nmax, MIN(nmax - 1, l)));
            }
            scale = sumlx / suml2; best = scale * sumlx;
            best_sumlx = sumlx; best_suml2 = suml2;
        }
        iscale = (nmax - 1 + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = q6_0_nearest_int_mul(iscale, x[i]);
            l = MAX(-nmax, MIN(nmax - 1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += (w * x[i]) * (float)l;
            suml2 += (w * (float)l) * (float)l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = q6_0_nearest_int_mul(iscale, x[i]);
                L[i] = (int8_t)(nmax + MAX(-nmax, MIN(nmax - 1, l)));
            }
            scale = sumlx / suml2; best = scale * sumlx;
            best_sumlx = sumlx; best_suml2 = suml2;
        }
    }

    sumlx = best_sumlx; suml2 = best_suml2;
    for (int iter = 0; iter < n * (2 * nmax - 1); ++iter) {
        float abs_gmax = 0, gmax = 0;
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            float w = qw ? qw[j] : rmse_type == 1 ? x[j] * x[j] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[j]) : sqrtf(fabsf(x[j]));
            int l = L[j] - nmax;
            float g = (scale * w) * std::fma(-scale, (float)l, x[j]);
            if ((g > 0 && l < nmax - 1) || (g < 0 && l > -nmax)) {
                float ag = fabsf(g);
                if (ag > abs_gmax) {
                    abs_gmax = ag; gmax = g; best_j = j;
                }
            }
        }
        if (best_j < 0) break;

        float new_sumlx = sumlx, new_suml2 = suml2;
        float w = qw ? qw[best_j] : rmse_type == 1 ? x[best_j] * x[best_j] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[best_j]) : sqrtf(fabsf(x[best_j]));
        int l = L[best_j] - nmax;
        if (gmax > 0) {
            new_sumlx += w * x[best_j];
            new_suml2 = std::fma(w, (float)(2 * l + 1), new_suml2);
            l += 1;
        } else {
            new_sumlx -= w * x[best_j];
            new_suml2 = std::fma(-w, (float)(2 * l - 1), new_suml2);
            l -= 1;
        }
        if (new_suml2 > 0 && new_sumlx * new_sumlx > best * new_suml2) {
            sumlx = new_sumlx; suml2 = new_suml2;
            scale = sumlx / suml2; best = scale * sumlx;
            L[best_j] = (int8_t)(l + nmax);
            GGML_ASSERT(L[best_j] >= 0 && L[best_j] <= 2 * nmax - 1);
        } else {
            break;
        }
    }
    return scale;
}

static void quantize_row_q6_0_impl(const float * GGML_RESTRICT x, block_q6_0 * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    static_assert(QK6_0 == 32, "QK6_0 must be 32");
    assert(n_per_row % QK6_0 == 0);

    float weight[QK6_0];
    int8_t L[QK6_0];

    float sigma2 = 0;
    if (quant_weights) {
        float sum_x2 = 0;
        for (int64_t j = 0; j < n_per_row; ++j) sum_x2 += x[j] * x[j];
        sigma2 = sum_x2 / n_per_row;
    }

    const int64_t nb = n_per_row / QK6_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + QK6_0 * ib;
        if (quant_weights) {
            const float * qw = quant_weights + QK6_0 * ib;
            for (int j = 0; j < QK6_0; ++j) weight[j] = qw[j] * sqrtf(std::fma(xb[j], xb[j], sigma2));
        } else {
            for (int j = 0; j < QK6_0; ++j) weight[j] = xb[j] * xb[j];
        }
        float d = q6_0_make_qx_quants(QK6_0, 32, xb, L, 1, weight);
        y[ib].d = GGML_FP32_TO_FP16(d);

        memset(y[ib].qh, 0, QK6_0 / 4);

        for (int j = 0; j < 16; ++j) {
            const uint8_t xi0 = (uint8_t)L[j];
            const uint8_t xi1 = (uint8_t)L[j + 16];
            y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            const uint8_t h = (uint8_t)((xi0 >> 4) | ((xi1 >> 4) << 2));
            y[ib].qh[j % 8] |= (uint8_t)(h << (4 * (j / 8)));
        }
    }
}

size_t ggml_quantize_q6_0_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK6_0 == 0);
    const size_t row_size = sizeof(block_q6_0) * (size_t)(n_per_row / QK6_0);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_q6_0_impl(src, (block_q6_0 *)qrow, n_per_row, imatrix);
        src += n_per_row;
        qrow += row_size;
    }
    return (size_t)nrows * row_size;
}

void quantize_row_q6_0_ref(const float * x, void * vy, int64_t k) {
    static const int qk = QK6_0;
    assert(k % qk == 0);

    block_q6_0 * y = (block_q6_0 *)vy;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -32;
        const float id = d ? 1.0f / d : 0.0f;

        memset(y[i].qh, 0, qk / 4);

        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = x[i * qk + 0      + j] * id;
            const float x1 = x[i * qk + qk / 2 + j] * id;
            const float w0 = x0 * x0;
            const float w1 = x1 * x1;

            const uint8_t xi0 = (uint8_t)MIN(63, (int8_t)(x0 + 32.5f));
            const uint8_t xi1 = (uint8_t)MIN(63, (int8_t)(x1 + 32.5f));

            y[i].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));

            const uint8_t h = (uint8_t)((xi0 >> 4) | ((xi1 >> 4) << 2));
            y[i].qh[j % (qk / 4)] |= (uint8_t)(h << (4 * (j / (qk / 4))));

            const float q0 = (float)xi0 - 32.f;
            const float q1 = (float)xi1 - 32.f;
            sumqx += w0 * x[i * qk + j] * q0 + w1 * x[i * qk + qk / 2 + j] * q1;
            sumq2 += w0 * q0 * q0 + w1 * q1 * q1;
        }
        y[i].d = sumq2 > 0 ? GGML_FP32_TO_FP16(sumqx / sumq2) : GGML_FP32_TO_FP16(d);
    }
}
