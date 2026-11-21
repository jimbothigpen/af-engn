#include "ggml-iq4-kt-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>

namespace {

// hsum_float_8 lane order (donor iqk_common.h)
static float sum8(const float * v) {
    return ((v[0]+v[4])+(v[2]+v[6]))+((v[1]+v[5])+(v[3]+v[7]));
}

// hsum_float_4x8 per-point order: (c0+c2)+(c1+c3).
// Verified empirically against the donor intrinsic (scope/iq4_kt/evidence/hsum4_order.c).
// NOT sequential and NOT sum8 -- IQ3_KT's sum8 is wrong for the group-4 path.
static float sum4(const float * v) {
    return (v[0]+v[2])+(v[1]+v[3]);
}

static int nearest_int_fma(float a, float b) {
    float val = std::fma(a, b, 12582912.f);
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static int bin5(float x) {
    return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
}

class QuantizerIQ4KT {
public:
    static constexpr int kNumVal = 32768, kGroupSize = 4, kNg = 8, kNblock = 8;
    static constexpr int kBlockSize = 32, kSuperBlockSize = 256;
    static constexpr int kNcluster = 625, kNneighbour = 6;

    std::vector<float> vals;
    std::vector<std::vector<int>> lists;
    std::vector<std::vector<float>> cvals;

    explicit QuantizerIQ4KT(int offset) : vals(kNumVal*kGroupSize), lists(kNcluster) {
        for (int i = 0; i < kNumVal; ++i) {
            uint32_t x = uint32_t(i + offset);
            for (int k = 0; k < kGroupSize; ++k) {
                x *= 0xCBAC1FEDu;
                const uint32_t s = x & 0x3f3f3f3fu;
                const int v = int(s & 255) + int((s >> 8) & 255) + int((s >> 16) & 255) + int(s >> 24) - 126;
                vals[kGroupSize*i + k] = (float)v;   // signed, kScale = 1.0f
            }
        }
        build_clusters();
    }

    const float * values() const { return vals.data(); }

private:
    void build_clusters() {
        // cluster_points: ndim==4 && ncluster==625 branch. Empty clusters keep a
        // ZERO centroid (donor prints a warning and continues -- it does NOT abort
        // on this branch). Bank 0 cluster 4 is empty; that is expected donor behaviour.
        float cent[kNcluster][kGroupSize] = {};
        int count[kNcluster] = {};
        for (int i = 0; i < kNumVal; ++i) {
            const float * vp = vals.data() + kGroupSize*i;
            int u = 0, s = 1;
            for (int k = 0; k < kGroupSize; ++k) { u += s*bin5(vp[k]); s *= 5; }
            assert(u >= 0 && u < kNcluster);
            ++count[u];
            for (int k = 0; k < kGroupSize; ++k) cent[u][k] += vp[k];
        }
        int nzero = 0;
        for (int c = 0; c < kNcluster; ++c) {
            if (!count[c]) { ++nzero; continue; }
            for (int k = 0; k < kGroupSize; ++k) cent[c][k] /= count[c];
        }
        if (nzero > 0) {
            fprintf(stderr, "cluster_points: %d out of %d clusters dir not have any points\n", nzero, kNcluster);
        }

        // finalize_clusters: kNneighbour nearest centroids per point, FMA dist2.
        std::vector<int> which(kNneighbour*kNumVal);
        for (int i = 0; i < kNumVal; ++i) {
            const float * vp = vals.data() + kGroupSize*i;
            float best[kNneighbour]; int ibest[kNneighbour];
            std::fill(best, best + kNneighbour, INFINITY);
            std::fill(ibest, ibest + kNneighbour, -1);
            for (int c = 0; c < kNcluster; ++c) {
                float dist2 = 0;
                for (int k = 0; k < kGroupSize; ++k) {
                    const float d = vp[k] - cent[c][k];
                    dist2 = std::fma(d, d, dist2);
                }
                for (int j = 0; j < kNneighbour; ++j) {
                    if (dist2 < best[j]) {
                        for (int k = kNneighbour - 1; k > j; --k) { best[k] = best[k-1]; ibest[k] = ibest[k-1]; }
                        best[j] = dist2; ibest[j] = c; break;
                    }
                }
            }
            for (int j = 0; j < kNneighbour; ++j) {
                assert(ibest[j] >= 0);
                lists[ibest[j]].push_back(i);
                which[kNneighbour*i + j] = ibest[j];
            }
        }

        // Pad every cluster to a multiple of 8 from the sorted nearest extras.
        std::vector<std::pair<float,int>> extra;
        extra.reserve(kNumVal);
        for (int c = 0; c < kNcluster; ++c) {
            auto & p = lists[c];
            if (!p.empty() && p.size() % 8 == 0) continue;
            extra.clear();
            for (int i = 0; i < kNumVal; ++i) {
                bool can_add = true;
                for (int j = 0; j < kNneighbour; ++j) {
                    if (which[kNneighbour*i + j] == c) { can_add = false; break; }
                }
                if (!can_add) continue;
                const float * vp = vals.data() + kGroupSize*i;
                float dist2 = 0;
                for (int k = 0; k < kGroupSize; ++k) {
                    const float d = vp[k] - cent[c][k];
                    dist2 = std::fma(d, d, dist2);
                }
                extra.emplace_back(dist2, i);
            }
            std::sort(extra.begin(), extra.end());
            const int nadd = 8*int((p.size() + 7)/8) - int(p.size());
            for (int j = 0; j < nadd; ++j) p.push_back(extra[j].second);
            assert(p.size() % 8 == 0);
        }

        cvals.resize(kNcluster);
        for (int c = 0; c < kNcluster; ++c) {
            cvals[c].resize(lists[c].size()*kGroupSize);
            float * ptr = cvals[c].data();
            for (int i : lists[c]) {
                std::memcpy(ptr, vals.data() + kGroupSize*i, kGroupSize*sizeof(float));
                ptr += kGroupSize;
            }
        }
    }

public:
    // Donor find_best_match, kGroupSize==4 AVX2 path (iqk_quantize.cpp:9941-10022).
    // ncluster==625 -> direct bin5 cluster select, NO cluster scan.
    // Neighbour search: plain muls vw*(diff*diff), NO sign mask (signed difference).
    // Eight candidate lanes per step; lane L holds point j+perm[L]; per-point score
    // uses sum4 order. Tie-break: first strictly smaller within a lane, then lanes 0..7.
    void find_best_match(float d, const float * x, const float * w, int * idx) const {
        if (!d) { std::memset(idx, 0, kNg*sizeof(int)); return; }
        const float id = 1/d;
        static const int perm[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        for (int g = 0; g < kNg; ++g) {
            float v[kGroupSize];
            int u = 0, s = 1;
            for (int k = 0; k < kGroupSize; ++k) { v[k] = id*x[kGroupSize*g + k]; u += s*bin5(v[k]); s *= 5; }
            const auto & p = lists[u];
            assert(!p.empty() && p.size() % 8 == 0);
            const float * cv = cvals[u].data();
            const int np = int(p.size());
            float best[8]; int bi[8];
            std::fill(best, best + 8, INFINITY);
            std::fill(bi, bi + 8, -1);
            for (int j = 0; j < np; j += 8) {
                for (int L = 0; L < 8; ++L) {
                    const float * q = cv + kGroupSize*(j + perm[L]);
                    float err[kGroupSize];
                    for (int k = 0; k < kGroupSize; ++k) {
                        const float diff = q[k] - v[k];
                        err[k] = w[kGroupSize*g + k]*(diff*diff);
                    }
                    const float score = sum4(err);
                    if (score < best[L]) { best[L] = score; bi[L] = j + perm[L]; }
                }
            }
            float b = INFINITY; int jb = -1;
            for (int L = 0; L < 8; ++L) if (best[L] < b) { b = best[L]; jb = bi[L]; }
            assert(jb >= 0);
            idx[g] = p[jb];
        }
    }

    // Donor find_best_scale AVX2: per-lane FMA over 8-wide vectors (two 4-groups
    // per vector), reduced with hsum_float_8.
    std::pair<float,float> find_best_scale(const float * x, const float * w, const int * idx) const {
        float sx[8] = {}, sq[8] = {};
        for (int l = 0; l < kBlockSize; l += 8) {
            for (int k = 0; k < 8; ++k) {
                const float q = vals[kGroupSize*idx[(l + k)/kGroupSize] + (l + k) % kGroupSize];
                const float qw = q*w[l + k];
                sx[k] = std::fma(qw, x[l + k], sx[k]);
                sq[k] = std::fma(qw, q, sq[k]);
            }
        }
        const float a = sum8(sx), b = sum8(sq);
        return b > 0 ? std::make_pair(a/b, a*a/b) : std::make_pair(0.f, 0.f);
    }

    static void set_weights(float ss, int nb, const float * x, const float * im, float * w, int nrow) {
        for (int b = 0; b < nb; ++b) {
            const float * xb = x + kSuperBlockSize*b;
            float * wb = w + kSuperBlockSize*b;
            const int n = std::min(kSuperBlockSize, nrow - kSuperBlockSize*b);
            for (int j = n; j < kSuperBlockSize; ++j) wb[j] = 0;
            float s = 0;
            for (int j = 0; j < n; ++j) s += xb[j]*xb[j];
            if (s < 1e-14f*n) { std::fill(wb, wb + n, 1e-4f); continue; }
            const float sigma = ss*s/n;
            if (im) {
                for (int g = 0; g < n/kBlockSize; ++g) {
                    float swx = 0, sw2 = 0, sx2 = 0;
                    for (int j = kBlockSize*g; j < kBlockSize*g + kBlockSize; ++j) {
                        wb[j] = im[kSuperBlockSize*b + j]*sqrtf(std::fma(xb[j], xb[j], sigma));
                        swx = std::fma(wb[j], std::abs(xb[j]), swx);
                        sw2 = std::fma(wb[j], wb[j], sw2);
                        sx2 = std::fma(xb[j], xb[j], sx2);
                    }
                    if (sx2 < 1e-14f || sw2 < 1e-14f || swx < 1e-14f) {
                        std::fill(wb + kBlockSize*g, wb + kBlockSize*g + kBlockSize, 1e-4f);
                    }
                }
            } else {
                for (int j = 0; j < n; ++j) wb[j] = std::fma(xb[j], xb[j], 0.25f*sigma);
            }
        }
    }
};

static const QuantizerIQ4KT & iq4kt_quantizer(bool with_offset) {
    static const QuantizerIQ4KT q1(4096);
    static const QuantizerIQ4KT q2(4096 + 32768);
    return with_offset ? q2 : q1;
}

void quantize_row_iq4_kt_impl(const float * x, void * vy, int n_per_row,
                              const float * quant_weights, float * all_scales, float * all_weights) {
    constexpr float kSigmaScale = 2.0f;
    constexpr int kNtry = 2;
    constexpr float fudge = 1.0f;
    using Q = QuantizerIQ4KT;

    float * dptr = (float *)vy;
    block_iq4_kt * y = (block_iq4_kt *)(dptr + 1);
    auto & quantizer1 = iq4kt_quantizer(false);
    auto & quantizer2 = iq4kt_quantizer(true);
    const int nblock = (n_per_row + Q::kSuperBlockSize - 1)/Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (!amax_row) {
        *dptr = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq4_kt));
        return;
    }

    int best_idx[2*Q::kNg];
    float xaux[Q::kBlockSize];
    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq4_kt));
        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                xaux[j] = xbl[ib*Q::kBlockSize + j];   // SIGNED, not abs
                amax = std::max(amax, std::abs(xaux[j]));
            }
            if (amax < 1e-16f) { scales[ib] = 0; continue; }
            float best = 0;
            const float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer1.find_best_match(amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) { best = score_p; scales[ib] = dp; }
                quantizer1.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) { best = score_m; scales[ib] = dm; }
            }
            quantizer2.find_best_match(scales[ib], xaux, weight, best_idx);
            auto [d2, score2] = quantizer2.find_best_scale(xaux, weight, best_idx);
            if (score2 > best) { best = score2; scales[ib] = d2; y[ibl].qs[ib] = 1; }
            bool with_offset = false;
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer2.find_best_match(amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) { best = score_p; scales[ib] = dp; with_offset = true; }
                quantizer2.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) { best = score_m; scales[ib] = dm; with_offset = true; }
            }
            if (with_offset) y[ibl].qs[ib] = 1;
            const float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) { amax_scale = abs_scale; max_scale = scales[ib]; }
        }
    }

    float d = -max_scale/64;
    *dptr = d * fudge;
    if (!d) return;

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    for (int iloop = 0; iloop < 1; ++iloop) {
        const float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto shb = y[ibl].qs;
            auto ql = (uint8_t *)(shb + Q::kNblock);
            auto qh = ql + kNumGroups;
            std::memset(qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                auto & quantizer = (y[ibl].qs[ib] & 1) ? quantizer2 : quantizer1;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) xaux[j] = xbl[ib*Q::kBlockSize + j];
                int ls = nearest_int_fma(id, scales[ib]);
                ls = std::min(ls, 63);                       // upper clamp ONLY
                *(uint8_t *)(shb + ib) = (uint8_t)(((ls + 64) << 1) | (shb[ib] & 1));
                const float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);
                for (int j = 0; j < Q::kNg; ++j) {
                    shb[ib] |= ((best_idx[j] >> 12) << (8 + 3*j));
                    ql[Q::kNg*ib + j] = best_idx[j] & 255;
                    qh[(Q::kNg*ib + j) % (kNumGroups/2)] |= ((best_idx[j] >> 8) & 0xf) << 4*((Q::kNg*ib + j)/(kNumGroups/2));
                    const float * xl = xaux + Q::kGroupSize*j;
                    const float * wl = weight + Q::kGroupSize*j;
                    const float * qv = quantizer.values() + Q::kGroupSize*best_idx[j];
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        const float q = qv[k]*ls;
                        sumqx = std::fma(wl[k]*xl[k], q, sumqx);
                        sumq2 = std::fma(wl[k]*q, q, sumq2);
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * fudge;
            if (!d) break;
        } else {
            break;
        }
    }
}

}

extern "C" size_t ggml_quantize_iq4_kt_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row > 0 && n_per_row % 32 == 0);
    const int64_t np = (n_per_row + 255)/256*256;
    const int64_t nb = n_per_row/256;
    const int64_t nt = (n_per_row % 256)/32;
    const size_t head = 4 + 128*nb;
    const size_t rs = (head + (nt ? 16*nt : 0) + 3)/4*4;
    std::vector<float> scales(np/32), weights(np);
    std::vector<float> xpad(np, 0.f), wpad;
    std::vector<char> ypad(4 + 128*(np/256));
    if (imatrix) { wpad.assign(imatrix, imatrix + n_per_row); wpad.resize(np, 0.f); }
    auto out = static_cast<char *>(dst);
    for (int64_t row = 0; row < nrows; ++row) {
        std::memcpy(xpad.data(), src, n_per_row*sizeof(float));
        // impl receives the ORIGINAL n_per_row (donor quantize_kt_rows lambda).
        quantize_row_iq4_kt_impl(xpad.data(), ypad.data(), (int)n_per_row,
                                 imatrix ? wpad.data() : nullptr, scales.data(), weights.data());
        std::memset(out, 0, rs);
        std::memcpy(out, ypad.data(), head);
        if (nt) {
            // repack_tail_iq4_kt: 16-byte records = qs[ib] | ql[8] | qh nibbles
            auto tail_block = reinterpret_cast<const block_iq4_kt *>(ypad.data() + 4) + nb;
            const uint8_t * ql = (const uint8_t *)(tail_block->qs + 8);
            const uint8_t * qh = ql + 64;
            uint8_t * tail = reinterpret_cast<uint8_t *>(out) + head;
            for (int ib = 0; ib < nt; ++ib) {
                uint8_t * record = tail + 16*ib;
                std::memcpy(record, tail_block->qs + ib, 4);
                std::memcpy(record + 4, ql + 8*ib, 8);
                uint8_t * qht = record + 12;
                std::memset(qht, 0, 4);
                for (int jj = 0; jj < 8; ++jj) {
                    const int j = 8*ib + jj;
                    qht[jj/2] |= ((qh[j % 32] >> 4*(j/32)) & 0xf) << 4*(jj & 1);
                }
            }
        }
        src += n_per_row;
        out += rs;
    }
    return nrows*rs;
}
