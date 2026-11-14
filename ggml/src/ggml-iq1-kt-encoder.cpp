#include "ggml-iq1-kt-encoder.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <utility>

namespace {
static const int8_t iq4k_values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
static int best_index_iq4nl(const int8_t * v, float x) {
    int i = 0;
    while (i < 15 && x >= v[i+1]) ++i;
    return i == 15 || x-v[i] < v[i+1]-x ? i : i+1;
}
static float sum8(const float * v) {
    return ((v[0]+v[4])+(v[2]+v[6]))+((v[1]+v[5])+(v[3]+v[7]));
}
class QuantizerIQ1KT {
public:
    static constexpr int kNumVal=8192, kGroupSize=8, kNg=4, kNblock=8, kBlockSize=32, kSuperBlockSize=256;
    std::vector<float> vals;
    std::vector<std::vector<int>> lists;
    float mid[8];
    QuantizerIQ1KT(): vals(8192*8), lists(256) {
        float lo[8], hi[8];
        std::fill(lo,lo+8,INFINITY); std::fill(hi,hi+8,-INFINITY);
        for (int i=0;i<8192;++i) {
            uint32_t x=i+4096;
            for (int k=0;k<8;++k) {
                x*=0xCBAC1FEDu; uint32_t s=x&0x3f3f3f3fu;
                float v=int(s&255)+int((s>>8)&255)+int((s>>16)&255)+int(s>>24)-126;
                vals[8*i+k]=v; lo[k]=std::min(lo[k],v); hi[k]=std::max(hi[k],v);
            }
        }
        for(int k=0;k<8;++k) mid[k]=0.5f*(lo[k]+hi[k]);
        float cent[256][8]={}; int count[256]={};
        for(int i=0;i<8192;++i) {
            int c=0; for(int k=0;k<8;++k) if(vals[8*i+k]>mid[k]) c|=1<<k;
            ++count[c]; for(int k=0;k<8;++k) cent[c][k]+=vals[8*i+k];
        }
        for(int c=0;c<256;++c) for(int k=0;k<8;++k) cent[c][k]/=count[c];
        auto distance=[&](int i,int c) {
            float s=0; for(int k=0;k<8;++k) { float d=vals[8*i+k]-cent[c][k]; s=std::fma(d,d,s); } return s;
        };
        for(int i=0;i<8192;++i) {
            float best[32]; int idx[32]; std::fill(best,best+32,INFINITY);
            for(int c=0;c<256;++c) {
                float d=distance(i,c);
                for(int j=0;j<32;++j) if(d<best[j]) {
                    for(int k=31;k>j;--k) {best[k]=best[k-1];idx[k]=idx[k-1];}
                    best[j]=d;idx[j]=c;break;
                }
            }
            for(int j=0;j<32;++j) lists[idx[j]].push_back(i);
        }
        for(int c=0;c<256;++c) {
            auto & p=lists[c]; if(p.size()%8==0) continue;
            std::vector<std::pair<float,int>> extra;
            for(int i=0;i<8192;++i) if(!std::binary_search(p.begin(),p.end(),i)) extra.emplace_back(distance(i,c),i);
            std::sort(extra.begin(),extra.end()); int n=8*((p.size()+7)/8)-p.size();
            for(int j=0;j<n;++j) p.push_back(extra[j].second);
        }
    }
    const float * values() const {return vals.data();}
    void find_best_match(float d,const float * x,const float * w,int * idx) const {
        if(!d) {std::memset(idx,0,4*sizeof(int));return;}
        float id=1/d;
        for(int g=0;g<4;++g) {
            float v[8]; int c=0;
            for(int k=0;k<8;++k) {v[k]=id*x[8*g+k];if(v[k]>mid[k]) c|=1<<k;}
            const auto & p=lists[c]; float best[8]; int bi[8];std::fill(best,best+8,INFINITY);
            int n=p.empty()?8192:int(p.size());
            for(int j=0;j<n;++j) {
                int qi=p.empty()?j:p[j];float err[8];
                for(int k=0;k<8;++k) {float diff=vals[8*qi+k]-v[k];err[k]=w[8*g+k]*(diff*diff);}
                float score=sum8(err);int lane=j%8;
                if(score<best[lane]) {best[lane]=score;bi[lane]=qi;}
            }
            float b=INFINITY;idx[g]=0;
            for(int k=0;k<8;++k) if(best[k]<b) {b=best[k];idx[g]=bi[k];}
        }
    }
    std::pair<float,float> find_best_scale(const float * x,const float * w,const int * idx) const {
        float sx[8]={},sq[8]={};
        for(int g=0;g<4;++g) for(int k=0;k<8;++k) {
            float q=vals[8*idx[g]+k], qw=q*w[8*g+k];
            sx[k]=std::fma(qw,x[8*g+k],sx[k]);sq[k]=std::fma(qw,q,sq[k]);
        }
        float a=sum8(sx),b=sum8(sq);
return b>0?std::make_pair(a/b,a*a/b):std::make_pair(0.f,0.f);
    }
    static void set_weights(float ss,int nb,const float * x,const float * im,float * w,int nrow) {
        for(int b=0;b<nb;++b) {
            const float * xb=x+256*b;float * wb=w+256*b;int n=std::min(256,nrow-256*b);float s=0;
            for(int j=0;j<n;++j) { float square=xb[j]*xb[j]; s+=square; }
            if(s<1e-14f*n) {std::fill(wb,wb+n,1e-4f);continue;}
            float sigma=ss*s/n;
            if(im) for(int g=0;g<n/32;++g) {
                float swx=0,sw2=0,sx2=0;
                for(int j=32*g;j<32*g+32;++j) {
                    wb[j]=im[256*b+j]*sqrtf(std::fma(xb[j],xb[j],sigma));
                    swx=std::fma(wb[j],std::abs(xb[j]),swx);sw2=std::fma(wb[j],wb[j],sw2);sx2=std::fma(xb[j],xb[j],sx2);
                }
                if(sx2<1e-14f||sw2<1e-14f||swx<1e-14f) std::fill(wb+32*g,wb+32*g+32,1e-4f);
            } else for(int j=0;j<n;++j) wb[j]=std::fma(xb[j],xb[j],0.25f*sigma);
        }
    }
};
static const QuantizerIQ1KT & iq1kt_quantizer() {static const QuantizerIQ1KT q;return q;}
static void pad_row(const float * src,float * dst,int64_t n) {
    int extra=((n+255)/256)*256-n;
    std::memcpy(dst,src,n*sizeof(float));
    if(n>=extra) std::memcpy(dst+n,src+n-extra,extra*sizeof(float));
    else for(int j=0;j<extra;j+=32) std::memcpy(dst+n+j,src+n-32,32*sizeof(float));
}
}

namespace {
void quantize_row_iq1_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ1KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq1_kt * y = (block_iq1_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq1kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq1_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = 0;
                continue;
            }
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg; score_p = score_m;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

            scale_0 -= 8;
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp1, score_p1] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm1, score_m1] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            if (score_p1 > score_p || score_m1 > score_p) {
                idx = best_idx;
                if (score_p1 > score_m1) scales[ib] = dp1;
                else {
                    scales[ib] = dm1; idx += Q::kNg;
                }
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale/iq4k_values[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry + iq4k_values[0])/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx = std::fma(wb[jj]*xb[jj],q,sumqx);
                        sumq2 = std::fma(wb[jj]*q,q,sumq2);
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    const float fudge = 1.07f;

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
            y[ibl].sh[ib] = ls;
        }
    }

    *dptr = fudge*d;
    if (!d) return;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[y[ibl].sh[ib] & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

                for (int j = 0; j < Q::kNg; ++j) {
                    y[ibl].ql[ib*Q::kNg+j] = best_idx[j] & 0xff;
                    y[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg+j] |= (((best_idx[j] >> 8) & 0xf) << 4*(ib/(Q::kNblock/2)));
                    y[ibl].sh[ib] |= ((best_idx[j] >> 12) << (4+j));
                    auto xl = xb + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx = std::fma(wl[k]*xl[k],q,sumqx);
                        sumq2 = std::fma(wl[k]*q,q,sumq2);
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * fudge;
            if (!d) return;
        } else {
            break;
        }

    }

}
}

extern "C" size_t ggml_quantize_iq1_kt_scalar(const float * src,void * dst,int64_t rows,int64_t n,const float * im) {
    assert(n>0 && n%32==0);
    int64_t np=(n+255)/256*256, nb=n/256, nt=n%256/32;
    size_t rs=(4+56*nb+(nt?8+4*nt+std::min<int64_t>(4*nt,16):0)+3)/4*4;
    std::vector<float> x(np),w(np),sc(np/32),iw;
    std::vector<int> idx(np/8);
    std::vector<uint8_t> tmp(4+56*(np/256));
    if(im) {iw.resize(np);pad_row(im,iw.data(),n);}
    for(int64_t row=0;row<rows;++row) {
        pad_row(src+row*n,x.data(),n);
        quantize_row_iq1_kt_impl(x.data(),tmp.data(),np,im?iw.data():nullptr,sc.data(),w.data(),idx.data());
        auto out=static_cast<uint8_t *>(dst)+row*rs;
        std::memset(out,0,rs);std::memcpy(out,tmp.data(),4+56*nb);
        if(nt) {
            auto tail=reinterpret_cast<const block_iq1_kt *>(tmp.data()+4)+nb;
            std::memcpy(out+4+56*nb,tail->sh,8);
            std::memcpy(out+12+56*nb,tail->ql,4*nt);
            std::memcpy(out+12+56*nb+4*nt,tail->qh,std::min<int64_t>(4*nt,16));
        }
    }
    return rows*rs;
}
