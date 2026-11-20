#include "ggml-iq3-kt-encoder.h"
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

static float sum8(const float * v) {
    return ((v[0]+v[4])+(v[2]+v[6]))+((v[1]+v[5])+(v[3]+v[7]));
}

static int nearest_int_fma(float a, float b) {
    float val = std::fma(a, b, 12582912.f);
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

class QuantizerIQ3KT {
public:
    static constexpr int kNumVal=65536, kGroupSize=8, kNg=4, kNblock=8, kBlockSize=32, kSuperBlockSize=256;
    std::vector<float> vals;
    std::vector<std::vector<int>> lists;
    std::vector<std::vector<float>> cvals;
    float mid[8];

    QuantizerIQ3KT(): vals(kNumVal*kGroupSize), lists(256) {
        for (int i=0;i<kNumVal;++i) {
            uint32_t x=i+4096;
            for (int k=0;k<8;++k) {
                x*=0xCBAC1FEDu; uint32_t s=x&0x3f3f3f3fu;
                int v=int(s&255)+int((s>>8)&255)+int((s>>16)&255)+int(s>>24)-126;
                vals[8*i+k]=std::abs(v);
            }
        }
        for (int k=0;k<8;++k) {
            std::vector<int> P(kNumVal);
            for (int i=0;i<kNumVal;++i) P[i]=(int)vals[8*i+k];
            std::sort(P.begin(),P.end());
            mid[k]=0.5f*(P[kNumVal/2]+P[kNumVal/2-1]);
        }
        float cent[256][8]={}; int count[256]={};
        for (int i=0;i<kNumVal;++i) {
            int c=0; for (int k=0;k<8;++k) if (vals[8*i+k]>mid[k]) c|=1<<k;
            ++count[c]; for (int k=0;k<8;++k) cent[c][k]+=vals[8*i+k];
        }
        for (int c=0;c<256;++c) {
            if (!count[c]) { fprintf(stderr,"IQ3_KT: cluster %d has no points\n",c); abort(); }
            for (int k=0;k<8;++k) cent[c][k]/=count[c];
        }
        std::vector<int> which(8*kNumVal);
        for (int i=0;i<kNumVal;++i) {
            float best[8]; int ibest[8];
            std::fill(best,best+8,INFINITY); std::fill(ibest,ibest+8,-1);
            for (int c=0;c<256;++c) {
                float s=0;
                for (int k=0;k<8;++k) { float d=vals[8*i+k]-cent[c][k]; s=std::fma(d,d,s); }
                for (int j=0;j<8;++j) if (s<best[j]) {
                    for (int k=7;k>j;--k) { best[k]=best[k-1]; ibest[k]=ibest[k-1]; }
                    best[j]=s; ibest[j]=c; break;
                }
            }
            for (int j=0;j<8;++j) { lists[ibest[j]].push_back(i); which[8*i+j]=ibest[j]; }
        }
        std::vector<std::pair<float,int>> extra;
        extra.reserve(kNumVal);
        for (int c=0;c<256;++c) {
            auto & p=lists[c];
            if (!p.empty() && p.size()%8==0) continue;
            extra.clear();
            for (int i=0;i<kNumVal;++i) {
                bool can_add=true;
                for (int j=0;j<8;++j) if (which[8*i+j]==c) { can_add=false; break; }
                if (!can_add) continue;
                float s=0;
                for (int k=0;k<8;++k) { float d=vals[8*i+k]-cent[c][k]; s=std::fma(d,d,s); }
                extra.emplace_back(s,i);
            }
            std::sort(extra.begin(),extra.end());
            int nadd=8*((p.size()+7)/8)-p.size();
            for (int j=0;j<nadd;++j) p.push_back(extra[j].second);
        }
        cvals.resize(256);
        for (int c=0;c<256;++c) {
            cvals[c].resize(lists[c].size()*8);
            float * ptr=cvals[c].data();
            for (int i : lists[c]) { std::memcpy(ptr,vals.data()+8*i,8*sizeof(float)); ptr+=8; }
        }
    }

    const float * values() const { return vals.data(); }

    void find_best_match(float d,const float * x,const float * w,int * idx) const {
        if (!d) { std::memset(idx,0,kNg*sizeof(int)); return; }
        float id=1/d;
        for (int g=0;g<kNg;++g) {
            float v[8]; int c=0;
            for (int k=0;k<8;++k) { v[k]=id*x[8*g+k]; if (v[k]>mid[k]) c|=1<<k; }
            const auto & p=lists[c];
            const float * cv=p.empty()?vals.data():cvals[c].data();
            int np=p.empty()?kNumVal:int(p.size());
            float best[8]; int bi[8];
            std::fill(best,best+8,INFINITY);
            for (int j=0;j<np;++j) {
                float err[8];
                for (int k=0;k<8;++k) { float diff=cv[8*j+k]-v[k]; err[k]=w[8*g+k]*(diff*diff); }
                float score=sum8(err); int lane=j%8;
                if (score<best[lane]) { best[lane]=score; bi[lane]=j; }
            }
            float b=INFINITY; int jb=-1;
            for (int k=0;k<8;++k) if (best[k]<b) { b=best[k]; jb=bi[k]; }
            idx[g]=p.empty()?jb:p[jb];
        }
    }

    std::pair<float,float> find_best_scale(const float * x,const float * w,const int * idx) const {
        float sx[8]={},sq[8]={};
        for (int g=0;g<kNg;++g) for (int k=0;k<8;++k) {
            float q=vals[8*idx[g]+k], qw=q*w[8*g+k];
            sx[k]=std::fma(qw,x[8*g+k],sx[k]); sq[k]=std::fma(qw,q,sq[k]);
        }
        float a=sum8(sx),b=sum8(sq);
        return b>0?std::make_pair(a/b,a*a/b):std::make_pair(0.f,0.f);
    }

    static void set_weights(float ss,int nb,const float * x,const float * im,float * w,int nrow) {
        for (int b=0;b<nb;++b) {
            const float * xb=x+256*b; float * wb=w+256*b;
            int n=std::min(256,nrow-256*b);
            for (int j=n;j<256;++j) wb[j]=0;
            float s=0;
            for (int j=0;j<n;++j) { float square=xb[j]*xb[j]; s+=square; }
            if (s<1e-14f*n) { std::fill(wb,wb+n,1e-4f); continue; }
            float sigma=ss*s/n;
            if (im) for (int g=0;g<n/32;++g) {
                float swx=0,sw2=0,sx2=0;
                for (int j=32*g;j<32*g+32;++j) {
                    wb[j]=im[256*b+j]*sqrtf(std::fma(xb[j],xb[j],sigma));
                    swx=std::fma(wb[j],std::abs(xb[j]),swx);
                    sw2=std::fma(wb[j],wb[j],sw2);
                    sx2=std::fma(xb[j],xb[j],sx2);
                }
                if (sx2<1e-14f||sw2<1e-14f||swx<1e-14f) std::fill(wb+32*g,wb+32*g+32,1e-4f);
            } else for (int j=0;j<n;++j) wb[j]=std::fma(xb[j],xb[j],0.25f*sigma);
        }
    }
};

static const QuantizerIQ3KT & iq3kt_quantizer() { static const QuantizerIQ3KT q; return q; }

void quantize_row_iq3_kt_impl(const float * x, void * vy, int n_per_row,
                              const float * quant_weights, float * all_scales, float * all_weights, float * qtmp) {
    constexpr float kSigmaScale = 2.0f;
    constexpr float kStep = 8.0f;
    constexpr float fudge = 1.0f;
    using Q = QuantizerIQ3KT;

    float * dptr = (float *)vy;
    block_iq3_kt * y = (block_iq3_kt *)(dptr + 1);
    int best_idx[2*Q::kNg];
    auto& quantizer = iq3kt_quantizer();
    int nblock = (n_per_row + 255)/256;

    float amax_row = 0;
    for (int j=0;j<n_per_row;++j) amax_row=std::max(amax_row,std::abs(x[j]));
    if (!amax_row) {
        *dptr = 0.f;
        std::memset(y,0,nblock*sizeof(block_iq3_kt));
        return;
    }

    Q::set_weights(kSigmaScale,nblock,x,quant_weights,all_weights,n_per_row);

    float amax_scale = 0, max_scale = 0;
    float xaux[Q::kBlockSize];

    for (int ibl=0;ibl<nblock;++ibl) {
        std::memset(&y[ibl],0,sizeof(block_iq3_kt));
        auto scales = all_scales + ibl*Q::kNblock;
        const float * xbl = x + ibl*Q::kSuperBlockSize;
        for (int ib=0;ib<Q::kNblock;++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j=0;j<Q::kBlockSize;++j) {
                float ax=std::abs(xb[j]); xaux[j]=ax; amax=std::max(amax,ax);
            }
            if (amax < 1e-16f) { scales[ib]=0.0f; continue; }

            float scale_0 = std::max(84.f, 123.f*amax/amax_row);
            float best = 0;
            bool found_solution = false;
            for (int itry=-3;itry<=3;++itry) {
                float scale = std::fma((float)itry,kStep,scale_0);
                quantizer.find_best_match(amax/scale,xaux,weight,best_idx);
                auto [d,score] = quantizer.find_best_scale(xaux,weight,best_idx);
                if (score > best) {
                    best = score;
                    found_solution = true;
                    scales[ib] = d;
                    std::memcpy(best_idx+Q::kNg,best_idx,Q::kNg*sizeof(int));
                }
            }
            if (!found_solution) {
                fprintf(stderr,"quantize_row_iq3_kt_impl: failed to find solution for a block\n");
                abort();
            }

            auto xt = qtmp + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            for (int ig=0;ig<Q::kNg;++ig) {
                auto q = quantizer.values() + Q::kGroupSize*best_idx[Q::kNg+ig];
                for (int j=0;j<Q::kGroupSize;++j) *xt++ = q[j];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) { amax_scale = abs_scale; max_scale = scales[ib]; }
        }
    }

    float d = max_scale/15.f;
    float best = 0;
    for (int itry=-9;itry<=9;++itry) {
        float id = std::fma((float)itry,0.2f,15.f)/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl=0;ibl<nblock;++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * qb = qtmp + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib=0;ib<Q::kNblock;++ib) {
                int ls = nearest_int_fma(id,scales[ib]);
                ls = std::max(0,std::min(15,ls));
                float dl = (float)ls;
                for (int j=0;j<Q::kBlockSize;++j) {
                    float q = dl*qb[j];
                    sumqx += wb[j]*std::abs(xb[j])*q;
                    sumq2 += wb[j]*q*q;
                }
                xb += Q::kBlockSize; wb += Q::kBlockSize; qb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) { d = sumqx/sumq2; best = d*sumqx; }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl=0;ibl<nblock;++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib=0;ib<Q::kNblock/2;++ib) {
            int ls1 = nearest_int_fma(id,scales[ib]);
            int ls2 = nearest_int_fma(id,scales[ib+Q::kNblock/2]);
            ls1 = std::max(0,std::min(15,ls1));
            ls2 = std::max(0,std::min(15,ls2));
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d * fudge;

    for (int iloop=0;iloop<1;++iloop) {
        float sumqx = 0, sumq2 = 0;
        for (int ibl=0;ibl<nblock;++ibl) {
            uint16_t * ql = (uint16_t *)y[ibl].ql;
            std::memset(y[ibl].qh,0,Q::kNblock*Q::kNg/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;
            for (int ib=0;ib<Q::kNblock;++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j=0;j<Q::kBlockSize;++j) {
                    xaux[j]=std::abs(xb[j]);
                    if (xb[j]<0) y[ibl].qh[j] |= (1 << ib);
                }
                int ls = (y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf;
                float dl = d*ls;
                quantizer.find_best_match(dl,xaux,weight,best_idx);
                for (int j=0;j<Q::kNg;++j) {
                    ql[ib*Q::kNg+j] = best_idx[j];
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto qv = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k=0;k<Q::kGroupSize;++k) {
                        float q = qv[k]*ls;
                        sumqx = std::fma(wl[k]*xl[k],q,sumqx);
                        sumq2 = std::fma(wl[k]*q,q,sumq2);
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

extern "C" size_t ggml_quantize_iq3_kt_scalar(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row > 0 && n_per_row%32 == 0);
    const int64_t np = (n_per_row+255)/256*256;
    const int64_t nb = n_per_row/256;
    const int64_t nt = (n_per_row%256)/32;
    const size_t head = 4 + 100*nb;
    const size_t rs = (head + (nt ? (nt+1)/2 + 12*nt : 0) + 3)/4*4;
    std::vector<float> scales(np/32), weights(np), qtmp(np);
    std::vector<float> xpad(np,0.f), wpad;
    std::vector<char> ypad(4 + 100*(np/256));
    if (imatrix) { wpad.assign(imatrix,imatrix+n_per_row); wpad.resize(np,0.f); }
    auto out = static_cast<char *>(dst);
    for (int64_t row=0;row<nrows;++row) {
        std::memcpy(xpad.data(),src,n_per_row*sizeof(float));
        quantize_row_iq3_kt_impl(xpad.data(),ypad.data(),n_per_row,imatrix?wpad.data():nullptr,
                                 scales.data(),weights.data(),qtmp.data());
        std::memset(out,0,rs);
        std::memcpy(out,ypad.data(),head);
        if (nt) {
            auto tail_block = reinterpret_cast<const block_iq3_kt *>(ypad.data()+4) + nb;
            uint8_t * tail = reinterpret_cast<uint8_t *>(out) + head;
            std::memcpy(tail,tail_block->ql,8*nt);
            uint8_t * qh = tail + 8*nt;
            std::memset(qh,0,4*nt);
            for (int ib=0;ib<nt;++ib) {
                for (int j=0;j<32;++j) {
                    if (tail_block->qh[j] & (1 << ib)) qh[4*ib+j/8] |= 1 << (j&7);
                }
            }
            uint8_t * st = tail + 12*nt;
            std::memset(st,0,(nt+1)/2);
            for (int ib=0;ib<nt;++ib) {
                st[ib/2] |= ((tail_block->scales[ib%4] >> 4*(ib/4)) & 0xf) << 4*(ib&1);
            }
        }
        src += n_per_row;
        out += rs;
    }
    return nrows*rs;
}