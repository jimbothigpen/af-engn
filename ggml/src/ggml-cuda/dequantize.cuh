#include "common.cuh"
#include "convert.cuh"

static __device__ __forceinline__ float iq1_kt_value(const void * row, int64_t i, int64_t width) {
    const int8_t scales[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    const int64_t b = i/256;
    const int nt = b < width/256 ? 8 : (width%256)/32;
    const int ib = (i%256)/32;
    const int ig = (i%32)/8;
    const int j = 4*ib+ig;
    const uint8_t * block = (const uint8_t *)row + 4 + 56*b;
    const uint8_t * ql = block+8;
    const uint8_t * qh = ql+4*nt;
    const int index = ql[j] | (((qh[j%16] >> (4*(j/16))) & 15) << 8) |
                      (((block[ib] >> (4+ig)) & 1) << 12);
    uint32_t state = index+4096;
    for (int k = 0; k <= i%8; ++k) {
        state *= 0xCBAC1FEDu;
    }
    const uint32_t bits = state & 0x3f3f3f3fu;
    const int q = int((bits & 255) + ((bits >> 8) & 255) + ((bits >> 16) & 255) + (bits >> 24)) - 126;
    float d;
    memcpy(&d, row, sizeof(d));
    return (d*scales[block[ib]&15])*q;
}

static __device__ __forceinline__ float iq2_kt_value(const void * row, int64_t i, int64_t width) {
const int8_t iq4k_values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
float d;
memcpy(&d, row, sizeof(d));
const int64_t nb = width/256;
const int64_t b = i/256;
float s;
uint32_t state;
if (b < nb) {
const uint8_t * block = (const uint8_t *)row + 4 + 68*b;
const uint16_t * ql = (const uint16_t *)(block + 4);
const int in_block = (int)(i%256);
if (in_block < 128) {
const int ib = in_block/32;
const int ig = (in_block%32)/8;
s = d*iq4k_values[block[ib] & 0xf]*1.05f;
state = (uint32_t)ql[4*ib + ig] + 4096u;
} else {
const int ib = (in_block - 128)/32;
const int ig = ((in_block - 128)%32)/8;
s = d*iq4k_values[block[ib] >> 4]*1.05f;
state = (uint32_t)ql[16 + 4*ib + ig] + 4096u;
}
} else {
const uint8_t * scales = (const uint8_t *)row + 4 + 68*nb;
const uint16_t * ql = (const uint16_t *)(scales + 4);
const int64_t it = i - nb*256;
const int ib = (int)(it/32);
const int ig = (int)((it%32)/8);
s = d*iq4k_values[(scales[ib%4] >> (4*(ib/4))) & 0xf]*1.05f;
state = (uint32_t)ql[4*ib + ig] + 4096u;
}
for (int k = 0; k <= (int)(i%8); ++k) {
state *= 0xCBAC1FEDu;
}
const uint32_t bits = state & 0x3f3f3f3fu;
const int q = int((bits & 255) + ((bits >> 8) & 255) + ((bits >> 16) & 255) + (bits >> 24)) - 126;
return s*q;
}

static __device__ __forceinline__ float iq3_kt_value(const void * row, int64_t i, int64_t width) {
float d;
memcpy(&d, row, sizeof(d));
d *= 1.01f;
const int64_t nb = width/256;
const int64_t b = i/256;
float s;
uint32_t state;
uint8_t sign_byte;
int sign_bit;
if (b < nb) {
const uint8_t * block = (const uint8_t *)row + 4 + 100*b;
const uint16_t * ql = (const uint16_t *)(block + 4);
const int in_block = (int)(i%256);
if (in_block < 128) {
const int ib = in_block/32;
const int ig = (in_block%32)/8;
s = d*(block[ib] & 0xf);
state = (uint32_t)ql[4*ib + ig] + 4096u;
sign_byte = block[68 + ig*8 + (i%8)];
sign_bit = ib;
} else {
const int ib = (in_block - 128)/32;
const int ig = ((in_block - 128)%32)/8;
s = d*(block[ib] >> 4);
state = (uint32_t)ql[16 + 4*ib + ig] + 4096u;
sign_byte = block[68 + ig*8 + (i%8)];
sign_bit = ib + 4;
}
} else {
const int64_t nt = (width%256)/32;
const uint8_t * tail = (const uint8_t *)row + 4 + 100*nb;
const uint16_t * ql = (const uint16_t *)tail;
const uint8_t * qh = tail + 8*nt;
const uint8_t * scales = tail + 12*nt;
const int64_t it = i - nb*256;
const int ib = (int)(it/32);
const int ig = (int)((it%32)/8);
s = d*((scales[ib/2] >> (4*(ib&1))) & 0xf);
state = (uint32_t)ql[4*ib + ig] + 4096u;
sign_byte = qh[4*ib + ig];
sign_bit = (int)(i%8);
}
for (int k = 0; k <= (int)(i%8); ++k) {
state *= 0xCBAC1FEDu;
}
const uint32_t bits = state & 0x3f3f3f3fu;
const int q = int((bits & 255) + ((bits >> 8) & 255) + ((bits >> 16) & 255) + (bits >> 24)) - 126;
const float v = (sign_byte & (1 << sign_bit)) ? -(float)abs(q) : (float)abs(q);
return s*v;
}

// IQ4_KT: 128 B per 256 block. qs bytes 0..8 are shb bank|ls|idx hi3; ql is 64 B at byte 32; qh is 32 B at byte 96.
// kScale = 1.0f, is_int=true; signed codebook, is_abs=false, 15-bit index, no sign plane.
static __device__ __forceinline__ float iq4_kt_value(const void * row, int64_t i, int64_t width) {
float d;
memcpy(&d, row, sizeof(d));
d *= 1.0f;
const int64_t nb = width/256;
const int64_t b = i/256;
float s;
uint32_t state;
uint32_t offset;
int ls;
if (b < nb) {
const uint8_t * block = (const uint8_t *)row + 4 + 128*b;
const uint32_t * shb = (const uint32_t *)block;
const uint8_t * ql = (const uint8_t *)(shb + 8);
const uint8_t * qh = ql + 64;
const int in_block = (int)(i%256);
const int ib = in_block/32;
const int ig = (in_block%32)/4;
const int jj = ib*8 + ig;
const uint32_t sh = shb[ib];
offset = (sh & 1) ? 4096u + 32768u : 4096u;
ls = (int)((sh & 0xff) >> 1) - 64;
s = d*(float)ls;
state = (uint32_t)ql[jj] + (((uint32_t)qh[jj%32] << (8 - 4*(jj/32))) & 0xf00u)
+ (((sh >> (8 + 3*ig)) & 7) << 12) + offset;
} else {
const uint8_t * tail = (const uint8_t *)row + 4 + 128*nb;
const int64_t it = i - nb*256;
const int ib = (int)(it/32);
const int ig = (int)((it%32)/4);
const uint32_t sh = *(const uint32_t *)(tail + 16*ib);
const uint8_t * ql = tail + 16*ib + 4;
const uint8_t * qh = tail + 16*ib + 12;
offset = (sh & 1) ? 4096u + 32768u : 4096u;
ls = (int)((sh & 0xff) >> 1) - 64;
s = d*(float)ls;
state = (uint32_t)ql[ig] + ((((uint32_t)qh[ig/2] >> (4*(ig&1))) & 0xf) << 8)
+ (((sh >> (8 + 3*ig)) & 7) << 12) + offset;
}
for (int k = 0; k <= (int)(i%4); ++k) {
state *= 0xCBAC1FEDu;
}
const uint32_t bits = state & 0x3f3f3f3fu;
const int q = int((bits & 255) + ((bits >> 8) & 255) + ((bits >> 16) & 255) + (bits >> 24)) - 126;
return s*(float)q;
}

static __device__ __forceinline__ float iq2_ks_value(const void * row, int64_t i) {
    const block_iq2_ks & b = ((const block_iq2_ks *) ((const char *) row + sizeof(half)))[i / QK_K];
    const int g = (i % QK_K) / 32;
    const int t = g / 2;
    const int h = g % 2;
    const int e = b.extra >> (2*t);
    const int scale = int(((b.scales[t] >> (4*h)) & 15) | ((e >> (4+h)) & 16)) - 16;
    const int q = (b.qs[(t/2)*32 + i%32] >> (4*(t%2) + 2*h)) & 3;
    const int values[8] = {-31, -13, 1, 17, -26, -8, 6, 22};
    return __half2float(*(const half *) row) * scale * values[((e >> h) & 1)*4 + q];
}

template<typename dst_t>
static __device__ void dequantize_iq2_ks(const void * row, const int64_t ib, dst_t * y, const int tid) {
    for (int j = tid; j < QK_K; j += 32) {
        y[j] = ggml_cuda_cast<dst_t>(iq2_ks_value(row, ib*QK_K + j));
    }
}


static __device__ __forceinline__ float iq2_kl_value(const void * row, int64_t i) {
    const float d = __half2float(*(const half *) row);
    const block_iq2_kl * x = (const block_iq2_kl *)((const char *) row + sizeof(half)) + (i / QK_K);
    const int in_block = (int)(i % QK_K);
    const int ib64 = in_block / 64;
    const int il = in_block % 64;

    const uint32_t sh = (uint32_t)(x->scales_h >> (4*ib64));
    float dl;
    int idx;
    int part;
    if (il < 32) {
        const int scl = (x->scales_l[(2*ib64+0)%4] >> (4*(ib64/2))) & 0xf;
        const int sch = sh & 3;
        dl = d * (float)((scl | (sch << 4)) - 32);
        const int j = il / 2;
        part = il % 2;
        idx = (x->qs[16*ib64 + j] & 0xf) | (((x->qh[j] >> (2*ib64+0)) & 1) << 4);
    } else {
        const int scl = (x->scales_l[(2*ib64+1)%4] >> (4*(ib64/2))) & 0xf;
        const int sch = (sh >> 2) & 3;
        dl = d * (float)((scl | (sch << 4)) - 32);
        const int j = (il - 32) / 2;
        part = (il - 32) % 2;
        idx = (x->qs[16*ib64 + j] >> 4) | (((x->qh[j] >> (2*ib64+1)) & 1) << 4);
    }

    const uint16_t entry = iq2kl_values[idx];
    const int8_t val = (part == 0) ? (int8_t)(entry & 0xff) : (int8_t)((entry >> 8) & 0xff);
    return dl * (float)val;
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_kl(const void * row, const int64_t ib, dst_t * y, const int tid) {
    for (int j = tid; j < QK_K; j += 32) {
        y[j] = ggml_cuda_cast<dst_t>(iq2_kl_value(row, ib*QK_K + j));
    }
}

static __device__ __forceinline__ float iq1_bn_value(const void * row, int64_t i) {
    half d16;
    memcpy(&d16, row, sizeof(d16));
    const float d = __half2float(d16);
    const block_iq1_bn * x = (const block_iq1_bn *)((const char *) row + sizeof(d16)) + (i / QK_IQ1BN);
    const int in_block = (int)(i % QK_IQ1BN);
    const int i16 = in_block / 16;
    const int r = in_block % 16;
    const uint8_t k_mult[5] = {81, 27, 9, 3, 1};
    uint8_t v;
    if (r < 15) {
        const uint8_t q = x->ql[3*i16 + (r / 5)];
        v = (uint8_t)(k_mult[r % 5] * q);
    } else {
        v = (uint8_t)(k_mult[i16] * x->extra);
    }
    const int8_t vs = (int8_t)((v + (v >> 1)) >> 7);
    return d * (float)(vs - 1);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_bn(const void * row, const int64_t ib, dst_t * y, const int tid) {
    for (int j = tid; j < QK_IQ1BN; j += 32) {
        y[j] = ggml_cuda_cast<dst_t>(iq1_bn_value(row, ib*QK_IQ1BN + j));
    }
}


static __device__ __forceinline__ float iq2_bn_value(const void * row, int64_t i) {
    float d;
    memcpy(&d, row, sizeof(d));
    const block_iq2_bn * x = (const block_iq2_bn *)((const char *) row + sizeof(d)) + (i / QK_IQ2BN);
    const int in_block = (int)(i % QK_IQ2BN);
    const int j = in_block % 16;
    const int k = in_block / 16;
    const int code = (x->qs[j] >> (2*k)) & 3;
    return d * (float)(code - 1);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_bn(const void * row, const int64_t ib, dst_t * y, const int tid) {
    for (int j = tid; j < QK_IQ2BN; j += 32) {
        y[j] = ggml_cuda_cast<dst_t>(iq2_bn_value(row, ib*QK_IQ2BN + j));
    }
}

static __device__ __forceinline__ float q6_0_value(const void * row, int64_t i) {
    const block_q6_0 * x = (const block_q6_0 *) row + (i / QK6_0);
    const int in_block = (int)(i % QK6_0);
    const int j  = in_block & 15;
    const int hi = in_block >> 4;
    const int h  = x->qh[j & 7] >> (4*(j >> 3));
    const int q  = hi ? ((x->qs[j] >> 4) | ((h << 2) & 0x30))
                      : ((x->qs[j] & 0x0F) | ((h << 4) & 0x30));
    return __half2float(x->d) * (float)(q - 32);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_0(const void * row, const int64_t ib, dst_t * y, const int tid) {
    for (int j = tid; j < QK6_0; j += 32) {
        y[j] = ggml_cuda_cast<dst_t>(q6_0_value(row, ib*QK6_0 + j));
    }
}

static __device__ __forceinline__ float pq2_0_value(const void * row, int64_t i) {
    const block_pq2_0 * x = (const block_pq2_0 *) row + (i / QK_PQ2_0);
    const int j = (int)(i % QK_PQ2_0);
    const int q = (x->qs[j >> 2] >> ((j & 3) * 2)) & 3;
    return (float)(q - 1) * __half2float(x->d);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_pq2_0(const void * row, const int64_t ib, dst_t * y, const int tid) {
    for (int j = tid; j < QK_PQ2_0; j += 32) {
        y[j] = ggml_cuda_cast<dst_t>(pq2_0_value(row, ib*QK_PQ2_0 + j));
    }
}

static __device__ __forceinline__ float iq3_ks_value(const void * row, int64_t i) {
const block_iq3_ks & b = ((const block_iq3_ks *) ((const char *) row + sizeof(half)))[i / QK_K];
const int e = i % QK_K;
const int t = e / 32;
const int j = e % 32;
const int ls = ((b.scales[t % 4] >> (4*(t / 4))) & 15) | (((b.extra >> t) & 1) << 4);
const int page = (b.extra >> (8 + t)) & 1;
const int q = ((b.qs[32*(t / 4) + j] >> (2*(t % 4))) & 3) | (((b.qh[j] >> t) & 1) << 2);
const int8_t values[16] = {-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51};
return __half2float(*(const half *) row) * (ls - 16) * values[8*page + q];
}

template<typename dst_t>
static __device__ void dequantize_iq3_ks(const void * row, const int64_t ib, dst_t * y, const int tid) {
for (int j = tid; j < QK_K; j += 32) {
y[j] = ggml_cuda_cast<dst_t>(iq3_ks_value(row, ib*QK_K + j));
}
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_ks(const void * row, const int64_t ib, dst_t * y, const int tid) {
const int8_t iq4k_values[32] = {
-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
-123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
};

float d;
memcpy(&d, row, sizeof(d));
const uint8_t * block = (const uint8_t *) row + sizeof(d) + ib*sizeof(block_iq4_ks);
const uint8_t * scales = block;
const uint8_t * qs = block + QK_K/32;

for (int g = tid; g < QK_K/32; g += 32) {
const uint8_t scale = scales[g];
const float dl = d * ((int) (scale & 254) - 127);
const int8_t * values = iq4k_values + ((scale & 1) << 4);
const uint8_t q = qs[16*g];
y[32*g +  0] = ggml_cuda_cast<dst_t>(dl * values[q & 0xf]);
y[32*g + 16] = ggml_cuda_cast<dst_t>(dl * values[q >> 4]);
for (int j = 1; j < 16; ++j) {
const uint8_t qj = qs[16*g + j];
y[32*g + j]      = ggml_cuda_cast<dst_t>(dl * values[qj & 0xf]);
y[32*g + 16 + j] = ggml_cuda_cast<dst_t>(dl * values[qj >> 4]);
}
}
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq5_ks(const void * row, const int64_t ib, dst_t * y, const int tid) {
const int8_t iq5nl_values[64] = {
-126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6,
-1, 5, 11, 17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121,
-124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10, -4,
1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89, 99, 111, 123,
};

float d;
memcpy(&d, row, sizeof(d));
const uint8_t * block = (const uint8_t *) row + 4 + 168*ib;
const uint8_t * scales = block;
const uint8_t * qs = block + 8;
const uint8_t * qh = block + 136;

for (int j = 0; j < 8; ++j) {
const int g = j/2;
const int h = j%2;
const uint8_t scale = scales[2*g + h];
const float dl = d * ((int) (scale & 254) - 127);
const int code = ((qs[32*g + tid] >> (4*h)) & 15) | (((qh[tid] >> (2*g + h)) & 1) << 4);
y[64*g + 32*h + tid] = ggml_cuda_cast<dst_t>(dl * iq5nl_values[((scale & 1) << 5) + code]);
}
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_kss(const void * row, const int64_t ib, dst_t * y, const int tid) {
    const int8_t iq4k_values[32] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
    };
    const uint8_t * bytes = (const uint8_t *) row;
    float d;
    memcpy(&d, row, sizeof(d));
    const uint8_t * block = bytes + 4 + 128*ib; // 4 + 128*N/256 row layout

    for (int g = tid; g < QK_K/32; g += 32) {
        const uint8_t * group = block + 16*g;
        int ls = 0;
        for (int r = 0; r < 8; ++r) {
            const uint16_t ur = uint16_t(group[2*r]) | (uint16_t(group[2*r + 1]) << 8);
            ls |= (ur & 1) << r;
        }
        const float dl = d * ((ls & 254) - 127);
        const int8_t * values = iq4k_values + ((ls & 1) << 4);
        for (int j = 0; j < 16; ++j) {
            const int lane = j/2;
            const uint16_t u = uint16_t(group[2*lane]) | (uint16_t(group[2*lane + 1]) << 8);
            const uint16_t aux = (u & 65534) ^ ((u & 65534) >> 1);
            const uint8_t qbyte = (aux >> (8*(j % 2))) & 255;
            y[32*g + j]      = ggml_cuda_cast<dst_t>(dl * values[qbyte & 15]);
            y[32*g + 16 + j] = ggml_cuda_cast<dst_t>(dl * values[qbyte >> 4]);
        }
    }
}

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const float d = x[ib].d;

    // Q2_0: 2 bits per element, 4 elements per byte.
    // Stored code c in {0,1,2,3} maps to symbol s = c - 1 in {-1, 0, +1, +2}.
    const int byte_index_0 = iqs / 4;
    const int bit_offset_0 = (iqs % 4) * 2;

    const int byte_index_1 = (iqs + 1) / 4;
    const int bit_offset_1 = ((iqs + 1) % 4) * 2;

    const int c0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 0x3;
    const int c1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 0x3;

    v.x = (c0 - 1) * d;
    v.y = (c1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

//================================== k-quants

// Each call dequantizes one super-block of QK_K values into y using the
// thread layout of the caller: 32 threads for q4_K, 64 threads otherwise.

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[ib].qs[32*n + l];
    dst_t * y = yy + 128*n;

    float dall = __low2half(x[ib].dm);
    float dmin = __high2half(x[ib].dm);
    y[l+ 0] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[ib].scales[is+0] >> 4));
    y[l+32] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[ib].scales[is+2] >> 4));
    y[l+64] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[ib].scales[is+4] >> 4));
    y[l+96] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[ib].scales[is+6] >> 4));
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = tid/4;
    const int64_t t = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(tid%4);
    const int64_t n = t / 4;
    const int64_t j = t - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is+0] >> 4) & 3) << 4) :
                          (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[ib].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + 128*n + 32*j;
    const uint8_t * q = x[ib].qs + 32*n;
    const uint8_t * hm = x[ib].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = ggml_cuda_cast<dst_t>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;

    // assume 32 threads
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + 64*il + n*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * q = x[ib].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = ggml_cuda_cast<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = ggml_cuda_cast<dst_t>(d2 * (q[l] >>  4) - m2);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + 64*il + 2*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + 128*ip + il;

    const float d = x[ib].d;

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = ggml_cuda_cast<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = ggml_cuda_cast<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = ggml_cuda_cast<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = ggml_cuda_cast<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

//================================== i-quants

// Each call dequantizes one super-block of QK_K values into y with 32
// threads; iq4_nl packs QK_K/QK4_NL sub-blocks per super-block.

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xxs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[ibs].qs[4*ib+il] | ((x[ibs].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_xxs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t  * q3 = x[ibs].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[ibs].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * qs = x[ibs].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[ibs].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[ibs].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[ibs].d * (1 + 2*((x[ibs].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[ibs].d * (2*((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_m(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq4_nl * x = (const block_iq4_nl *) vx + ibs*(QK_K/QK4_NL);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_k(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const int values[32] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117
    };
    const block_iq4_k & x = ((const block_iq4_k *) vx)[ibs];
    const float d = __half2float(x.d);
    for (int j = tid; j < QK_K; j += 32) {
        const int group = j / 32;
        const int half = (j % 32) / 16;
        const int lane = j % 16;
        const int high = (x.scales_h[group / 2] >> (4 * (group % 2))) & 15;
        const int low = (x.scales_l[group] >> (4 * half)) & 15;
        const int scale = (low | (((high >> (2 * half)) & 3) << 4)) - 32;
        const int table = ((x.extra >> (2 * group + half)) & 1) * 16;
        const int code = (x.qs[16 * group + lane] >> (4 * half)) & 15;
        const float dl = d * scale;
        yy[j] = ggml_cuda_cast<dst_t>(dl * values[table + code]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_k(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const int8_t values[8] = { -31, -13, 1, 17, -26, -8, 6, 22 };
    const block_iq2_k & x = ((const block_iq2_k *) vx)[ibs];
    const float d = __half2float(x.d);

    for (int j = tid; j < QK_K; j += 32) {
        const int ib32 = j / 32;
        const int half = (j % 32) / 16;
        const int lane = j % 16;
        const int scale = ((half == 0 ? (x.scales[ib32] & 0xf) : (x.scales[ib32] >> 4)) - 8);
        const int val_offset = ((x.extra >> (2 * ib32 + half)) & 1) ? 4 : 0;
        const int qs_idx = (ib32 < 4 ? 0 : 32) + (half == 0 ? lane : 16 + lane);
        const int shift = 2 * (ib32 % 4);
        const int code = (x.qs[qs_idx] >> shift) & 3;
        const float dl = d * scale;
        yy[j] = ggml_cuda_cast<dst_t>(dl * values[val_offset + code]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq5_k(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const int8_t values[64] = {
        -126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6, -1, 5, 11, 17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121,
        -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10, -4,  1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89, 99, 111, 123,
    };
    const block_iq5_k & x = ((const block_iq5_k *) vx)[ibs];
    const float d = __half2float(x.d);

    for (int j = tid; j < QK_K; j += 32) {
        const int ib64 = j / 64;
        const int sub16 = (j % 64) / 16;
        const int lane = j % 16;
        const uint8_t sh = x.scales_h[ib64];
        const uint8_t sl = x.scales_l[2*ib64 + sub16/2];
        const int raw_l = (sub16 % 2 == 0) ? (sl & 0xf) : (sl >> 4);
        const int raw_h = (sub16 == 0) ? ((sh << 4) & 0x30) :
                          (sub16 == 1) ? ((sh << 2) & 0x30) :
                          (sub16 == 2) ? ((sh     ) & 0x30) :
                                         ((sh >> 2) & 0x30);
        const int scale = (raw_l | raw_h) - 32;
        const int val_offset = ((x.extra >> (4 * ib64 + sub16)) & 1) ? 32 : 0;
        const int qs_idx = 32 * ib64 + (sub16 % 2 == 0 ? lane : 16 + lane);
        const int qh_idx = (sub16 % 2 == 0 ? lane : 16 + lane);
        const int shift_qh = 2 * ib64;
        const int low4 = (sub16 < 2) ? (x.qs[qs_idx] & 0xf) : (x.qs[qs_idx] >> 4);
        const int high1 = (sub16 < 2) ? ((x.qh[qh_idx] >> shift_qh) & 1) : (((x.qh[qh_idx] >> shift_qh) & 2) >> 1);
        const int code = low4 | (high1 << 4);
        yy[j] = ggml_cuda_cast<dst_t>(d * scale * values[val_offset + code]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq6_k(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const int8_t values[128] = {
        -127, -121, -115, -109, -104,  -98,  -93,  -88,  -84,  -79,  -74,  -70,  -66,  -62,  -58,  -54,
         -51,  -47,  -44,  -40,  -37,  -34,  -31,  -28,  -25,  -22,  -19,  -16,  -13,  -11,   -8,   -5,
          -2,    0,    3,    6,    9,   12,   14,   17,   20,   23,   27,   30,   33,   36,   40,   44,
          47,   51,   55,   59,   63,   68,   72,   77,   82,   87,   92,   98,  103,  109,  115,  121,
        -126, -120, -114, -108, -103,  -97,  -92,  -87,  -83,  -78,  -73,  -69,  -65,  -61,  -57,  -53,
         -50,  -46,  -43,  -39,  -36,  -33,  -30,  -27,  -24,  -21,  -18,  -15,  -12,  -10,   -7,   -4,
          -1,    1,    4,    7,   10,   13,   15,   18,   21,   24,   28,   31,   34,   37,   41,   45,
          48,   52,   56,   60,   64,   69,   73,   78,   83,   88,   93,   99,  104,  110,  116,  122,
    };
    const block_iq6_k & x = ((const block_iq6_k *) vx)[ibs];
    const float d = __half2float(x.d);

    for (int j = tid; j < QK_K; j += 32) {
        const int ib64 = j / 64;
        const int quarter = (j % 64) / 16;
        const int lane = j % 16;
        const float dl = d * (float)x.scales[4*ib64 + quarter];
        const int val_offset = ((x.extra >> (4*ib64 + quarter)) & 1) ? 64 : 0;
        const int use_upper = quarter >> 1;
        const int qs_base = 32*ib64 + (quarter & 1)*16 + lane;
        const int qh_base = (ib64 >> 1)*32 + (quarter & 1)*16 + lane;
        const int qh_shift = (ib64 & 1)*4;
        const uint8_t qs = x.qs[qs_base];
        const uint8_t qh = x.qh[qh_base];
        const int code = use_upper ?
            ((qs >> 4) | (((qh >> qh_shift) & 0x0c) << 2)) :
            ((qs & 0xf) | (((qh >> qh_shift) & 0x03) << 4));
        yy[j] = ggml_cuda_cast<dst_t>(dl * values[val_offset + code]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_k(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const int8_t values[16] = { -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51 };
    const block_iq3_k & x = ((const block_iq3_k *) vx)[ibs];
    const float d = __half2float(x.d);

    for (int j = tid; j < QK_K; j += 32) {
        const int ib32 = j / 32;
        const int half = (j % 32) / 16;
        const int lane = j % 16;
        const int raw = half == 0 ? (x.scales_l[ib32] & 0xf) : (x.scales_l[ib32] >> 4);
        const int scale = (2 * raw + 1) * (((x.scales_h >> (2 * ib32 + half)) & 1) ? -1 : 1);
        const int val_offset = ((x.extra >> (2 * ib32 + half)) & 1) ? 8 : 0;
        const int off = (ib32 < 4 ? 0 : 32) + 16 * half + lane;
        const int code = ((x.qs[off] >> (2 * (ib32 % 4))) & 3) | (((x.qh[16 * half + lane] >> ib32) & 1) << 2);
        yy[j] = ggml_cuda_cast<dst_t>(d * scale * values[val_offset + code]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ibs].qs + 16*ib + 4*il;
    const float d = (float)x[ibs].d * ((((x[ibs].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[ibs].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_mxfp4(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_mxfp4 * x = (const block_mxfp4 *) vx + ibs*(QK_K/QK_MXFP4);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] & 0xf]*0.5f);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] >>  4]*0.5f);
    }
}
