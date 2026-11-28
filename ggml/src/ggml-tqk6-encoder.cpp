#include "ggml-tqk6-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-tqk6-lut.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <mutex>

#define TQK6_NSEP   512
#define TQK6_NSTATE 32768
#define TQK6_STEPS  32
#define TQK6_INF    1e30f

static float tqk6_cb[TQK6_NSTATE][4];
static float tqk6_cb_sq[TQK6_NSTATE][4];
static std::once_flag tqk6_cb_once;

static void tqk6_cb_init() {
    for (uint32_t s = 0; s < TQK6_NSTATE; ++s) {
        const uint32_t z    = s * 0x9e3779b1u;
        const uint32_t idx0 = z >> 21;
        const uint32_t idx1 = (z >> 10) & 2047u;
        tqk6_cb[s][0] = GGML_FP16_TO_FP32(tqk6_lut_f16[2*idx0 + 0]);
        tqk6_cb[s][1] = GGML_FP16_TO_FP32(tqk6_lut_f16[2*idx0 + 1]);
        tqk6_cb[s][2] = GGML_FP16_TO_FP32(tqk6_lut_f16[2*idx1 + 0]);
        tqk6_cb[s][3] = GGML_FP16_TO_FP32(tqk6_lut_f16[2*idx1 + 1]);
        for (int m = 0; m < 4; ++m) {
            tqk6_cb_sq[s][m] = tqk6_cb[s][m] * tqk6_cb[s][m];
        }
    }
}

static void tqk6_encode_block(const float * x, const float * imatrix, block_tqk6 * y) {
    std::call_once(tqk6_cb_once, tqk6_cb_init);

    float  xs[QK_TQK];
    double w[QK_TQK];
    double sw = 0.0;
    for (int j = 0; j < QK_TQK; ++j) {
        xs[j] = isfinite(x[j]) ? x[j] : 0.0f;
        double wj = 1.0;
        if (imatrix) {
            const double v = (double) imatrix[j];
            if (isfinite(v) && v > 0.0) {
                wj = v;
            }
        }
        w[j] = wj;
        sw  += wj;
    }
    const double wnorm = (double) QK_TQK / sw;
    float wf[QK_TQK];
    for (int j = 0; j < QK_TQK; ++j) {
        wf[j] = (float) (w[j] * wnorm);
    }

    double e2 = 0.0;
    for (int j = 0; j < QK_TQK; ++j) {
        e2 += (double) wf[j] * (double) xs[j] * (double) xs[j];
    }
    memset(y, 0, sizeof(*y));
    if (!(e2 > 0.0) || !isfinite(e2)) {
        return;
    }

    double d0 = sqrt(e2 / (double) QK_TQK) / 1.0722;
    if (d0 > 65504.0) {
        d0 = 65504.0;
    }
    if (d0 < 6.103515625e-5) {
        d0 = 6.103515625e-5;
    }
    const float d_init = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16((float) d0));
    const float invd   = 1.0f / d_init;

    float tx[QK_TQK];
    for (int j = 0; j < QK_TQK; ++j) {
        float v = xs[j] * invd;
        if (v >  1.0e4f) v =  1.0e4f;
        if (v < -1.0e4f) v = -1.0e4f;
        tx[j] = v;
    }

    static thread_local float   cost[TQK6_NSEP];
    static thread_local float   ncost[TQK6_NSEP];
    static thread_local uint8_t prev_lo[TQK6_STEPS][TQK6_NSEP];

    float a[TQK6_STEPS][4];
    float wk[TQK6_STEPS][4];
    for (int k = 0; k < TQK6_STEPS; ++k) {
        const int t = 31 - k;
        for (int m = 0; m < 4; ++m) {
            wk[k][m] = wf[4*t + m];
            a[k][m]  = -2.0f * wf[4*t + m] * tx[4*t + m];
        }
    }

    for (int i = 0; i < TQK6_NSEP; ++i) {
        cost[i] = 0.0f;
    }
    for (int k = 0; k < TQK6_STEPS; ++k) {
        const float * ak = a[k];
        const float * wq = wk[k];
        for (int ns = 0; ns < TQK6_NSEP; ++ns) {
            float best = TQK6_INF;
            const int base = ns << 6;
            for (int lo = 0; lo < 64; ++lo) {
                const int s = base | lo;
                const float e = wq[0]*tqk6_cb_sq[s][0] + wq[1]*tqk6_cb_sq[s][1] + wq[2]*tqk6_cb_sq[s][2] + wq[3]*tqk6_cb_sq[s][3]
                              + ak[0]*tqk6_cb[s][0] + ak[1]*tqk6_cb[s][1] + ak[2]*tqk6_cb[s][2] + ak[3]*tqk6_cb[s][3];
                const float tot = cost[s & 511] + e;
                if (tot < best) {
                    best = tot;
                }
            }
            ncost[ns] = best;
        }
        memcpy(cost, ncost, sizeof(cost));
    }
    int sep0 = 0;
    for (int i = 1; i < TQK6_NSEP; ++i) {
        if (cost[i] < cost[sep0]) {
            sep0 = i;
        }
    }

    for (int i = 0; i < TQK6_NSEP; ++i) {
        cost[i] = TQK6_INF;
    }
    cost[sep0] = 0.0f;
    for (int k = 0; k < TQK6_STEPS; ++k) {
        const float * ak = a[k];
        const float * wq = wk[k];
        for (int ns = 0; ns < TQK6_NSEP; ++ns) {
            float best = TQK6_INF;
            uint8_t blo = 0;
            const int base = ns << 6;
            for (int lo = 0; lo < 64; ++lo) {
                const int s = base | lo;
                const float cp = cost[s & 511];
                if (cp >= TQK6_INF * 0.5f) {
                    continue;
                }
                const float e = wq[0]*tqk6_cb_sq[s][0] + wq[1]*tqk6_cb_sq[s][1] + wq[2]*tqk6_cb_sq[s][2] + wq[3]*tqk6_cb_sq[s][3]
                              + ak[0]*tqk6_cb[s][0] + ak[1]*tqk6_cb[s][1] + ak[2]*tqk6_cb[s][2] + ak[3]*tqk6_cb[s][3];
                const float tot = cp + e;
                if (tot < best) {
                    best = tot;
                    blo  = (uint8_t) lo;
                }
            }
            ncost[ns]      = best;
            prev_lo[k][ns] = blo;
        }
        memcpy(cost, ncost, sizeof(cost));
    }

    uint16_t states[TQK6_STEPS];
    int ns = sep0;
    for (int k = TQK6_STEPS - 1; k >= 0; --k) {
        const int s = (ns << 6) | prev_lo[k][ns];
        states[k]   = (uint16_t) s;
        ns          = s & 511;
    }
    assert(ns == sep0);

    double num = 0.0;
    double den = 0.0;
    for (int k = 0; k < TQK6_STEPS; ++k) {
        const int t = 31 - k;
        const int s = states[k];
        for (int m = 0; m < 4; ++m) {
            const double wm = (double) wk[k][m];
            num += wm * (double) xs[4*t + m] * (double) tqk6_cb[s][m];
            den += wm * (double) tqk6_cb_sq[s][m];
        }
    }
    double d_ref = den > 0.0 ? num / den : 0.0;
    if (d_ref >  65504.0) d_ref =  65504.0;
    if (d_ref < -65504.0) d_ref = -65504.0;
    y->d = GGML_FP32_TO_FP16((float) d_ref);

    for (int k = 0; k < TQK6_STEPS; ++k) {
        const uint32_t sym = states[k] & 63u;
        const int bit = 6 * k;
        const int bi  = bit >> 3;
        const int sh  = bit & 7;
        y->qs[bi] |= (uint8_t) (sym << sh);
        if (sh > 2) {
            y->qs[bi + 1] |= (uint8_t) (sym >> (8 - sh));
        }
    }
}

void quantize_row_tqk6_ref(const float * x, void * vy, int64_t k) {
    assert(k % QK_TQK == 0);
    const int64_t nb = k / QK_TQK;
    block_tqk6 * y = (block_tqk6 *) vy;
    for (int64_t i = 0; i < nb; ++i) {
        tqk6_encode_block(x + i*QK_TQK, nullptr, y + i);
    }
}

size_t ggml_quantize_tqk6_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const int64_t nb = n_per_row / QK_TQK;
    const size_t row_size = (size_t) nb * sizeof(block_tqk6);
    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrows; ++row) {
        block_tqk6 * y = (block_tqk6 *) qrow;
        for (int64_t i = 0; i < nb; ++i) {
            tqk6_encode_block(src + i*QK_TQK, imatrix ? imatrix + i*QK_TQK : nullptr, y + i);
        }
        src  += n_per_row;
        qrow += row_size;
    }
    return (size_t) nrows * row_size;
}
