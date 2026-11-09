#if !defined(DATA_A_F32) && !defined(DATA_A_F16)
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#endif

#include "types.glsl"

#if defined(DATA_A_F32)
FLOAT_TYPE dequantize1(uint ib, uint iqs, uint a_offset) {
    return data_a[a_offset + ib];
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(data_a[a_offset + ib    ], data_a[a_offset + ib + 1],
                data_a[a_offset + ib + 2], data_a[a_offset + ib + 3]);
}
vec4 dequantize4_2aligned(uint ib, uint iqs, uint a_offset) {
    const vec2 a = data_a_packed64[(a_offset + ib)/2];
    const vec2 b = data_a_packed64[(a_offset + ib)/2 + 1];
    return vec4(a, b);
}

#endif

#if defined(DATA_A_F16)
FLOAT_TYPE dequantize1(uint ib, uint iqs, uint a_offset) {
    return data_a[a_offset + ib];
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(data_a[a_offset + ib    ], data_a[a_offset + ib + 1],
                data_a[a_offset + ib + 2], data_a[a_offset + ib + 3]);
}
vec4 dequantize4_2aligned(uint ib, uint iqs, uint a_offset) {
    const vec2 a = data_a_packed32[(a_offset + ib)/2];
    const vec2 b = data_a_packed32[(a_offset + ib)/2 + 1];
    return vec4(a, b);
}
#endif

#if defined(DATA_A_BF16)
FLOAT_TYPE dequantize1(uint ib, uint iqs, uint a_offset) {
    return bf16_to_fp32(data_a[a_offset + ib]);
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(bf16_to_fp32(data_a[a_offset + ib]), bf16_to_fp32(data_a[a_offset + ib + 1]));
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(bf16_to_fp32(data_a[a_offset + ib    ]), bf16_to_fp32(data_a[a_offset + ib + 1]),
                bf16_to_fp32(data_a[a_offset + ib + 2]), bf16_to_fp32(data_a[a_offset + ib + 3]));
}
vec4 dequantize4_2aligned(uint ib, uint iqs, uint a_offset) {
    const uint a = data_a_packed32[(a_offset + ib)/2];
    const uint b = data_a_packed32[(a_offset + ib)/2 + 1];
    return vec4(uintBitsToFloat((a & 0x0000ffff) << 16),
                uintBitsToFloat( a & 0xffff0000),
                uintBitsToFloat((b & 0x0000ffff) << 16),
                uintBitsToFloat( b & 0xffff0000));
}
#endif

#if defined(DATA_A_Q4_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2(vui & 0xF, vui >> 4) - 8.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f);
}
#endif

#if defined(DATA_A_Q4_1)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(vui & 0xF, vui >> 4);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12);
}
#endif

#if defined(DATA_A_Q5_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = uint(data_a[a_offset + ib].qh[1]) << 16 | data_a[a_offset + ib].qh[0];
    const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y) - 16.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = uint(data_a_packed16[a_offset + ib].qh[1]) << 16 | data_a_packed16[a_offset + ib].qh[0];
    const ivec2 qh0 = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f);
}
#endif

#if defined(DATA_A_Q5_1)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a[a_offset + ib].qh;
    const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a_packed16[a_offset + ib].qh;
    const ivec2 qh0 = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y);
}
#endif

#if defined(DATA_A_Q8_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(int(data_a[a_offset + ib].qs[iqs]), int(data_a[a_offset + ib].qs[iqs + 1]));
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2])).xy; // vec4 used due to #12147
    const i8vec2 v1 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2 + 1])).xy;
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
#endif

#if defined(DATA_A_Q1_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);
    return vec2(
        (bits & 1u) != 0u ? 1.0f : -1.0f,
        (bits & 2u) != 0u ? 1.0f : -1.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);
    return vec4(
        (bits & 1u) != 0u ? 1.0f : -1.0f,
        (bits & 2u) != 0u ? 1.0f : -1.0f,
        (bits & 4u) != 0u ? 1.0f : -1.0f,
        (bits & 8u) != 0u ? 1.0f : -1.0f);
}
#endif

#if defined(DATA_A_Q2_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 4u]) >> (2u * (iqs % 4u));
    return vec2(bits & 3u, (bits >> 2u) & 3u) - 1.0f;
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 4u]);
    return vec4(bits & 3u, (bits >> 2u) & 3u, (bits >> 4u) & 3u, bits >> 6u) - 1.0f;
}
#endif

#if defined(DATA_A_IQ1_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;
    const int i8 = int(iqs % 8);
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const float dl = float(2 * bitfieldExtract(qh, 12, 3) + 1);
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint idxhi = bitfieldExtract(qh, 3 * int(ib8 & 3), 3);
    const int16_t grid = int16_t(iq1s_grid[qs | (idxhi << 8)]);
    // Signed bitfield extract.
    const ivec2 gvec = ivec2(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2)
    );
    return dl * (vec2(gvec) + delta);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;
    const int i8 = int(iqs % 8);
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const float dl = 2 * bitfieldExtract(qh, 12, 3) + 1;
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)]);
    // Signed bitfield extract.
    const ivec4 gvec = ivec4(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2),
      bitfieldExtract(grid, 2 * (i8 + 2), 2),
      bitfieldExtract(grid, 2 * (i8 + 3), 2)
    );
    return dl * (vec4(gvec) + delta);
}
#endif

#if defined(DATA_A_IQ1_M)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib8 = iqs / 8;
    const uint ib16 = iqs / 16;
    const int i8 = int(iqs % 8);
    const uint sc = data_a[a_offset + ib].scales[iqs / 64];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);
    // Signed bitfield extract.
    const ivec2 gvec = ivec2(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2)
    );
    return dl * (vec2(gvec) + delta);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib8 = iqs / 8;
    const uint ib16 = iqs / 16;
    const int i8 = int(iqs % 8);
    const uint sc = data_a[a_offset + ib].scales[iqs / 64];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);
    // Signed bitfield extract.
    const ivec4 gvec = ivec4(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2),
      bitfieldExtract(grid, 2 * (i8 + 2), 2),
      bitfieldExtract(grid, 2 * (i8 + 3), 2)
    );
    return dl * (vec4(gvec) + delta);
}
#endif

#if defined(DATA_A_IQ2_XXS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = (iqs / 8) % 4;
    const uint qs = data_a[a_offset + ib].qs[8 * ib32 + ib8];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[4 * ib32 + 2],
        data_a_packed16[a_offset + ib].qs[4 * ib32 + 3]));
    const float db = 0.25 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xxs_grid[qs][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = (iqs / 8) % 4;
    const uint qs = data_a[a_offset + ib].qs[8 * ib32 + ib8];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[4 * ib32 + 2],
        data_a_packed16[a_offset + ib].qs[4 * ib32 + 3]));
    const float db = 0.25 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xxs_grid[qs][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ2_XS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint scale = (data_a[a_offset + ib].scales[iqs / 32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[iqs / 8];
    const float db = 0.25 * (0.5 + scale);
    const uint sign7 = qs >> 9;
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xs_grid[qs & 511][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint scale = (data_a[a_offset + ib].scales[iqs / 32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[iqs / 8];
    const float db = 0.25 * (0.5 + scale);
    const uint sign7 = qs >> 9;
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xs_grid[qs & 511][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ2_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;

    const uint scale = (data_a[a_offset + ib].scales[ib32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qhshift = 2 * (ib8 % 4);
    const uint sign = data_a[a_offset + ib].qs[QUANT_K / 8 + ib8] >> (iqs % 8);

    const float db = 0.25 * (0.5 + scale);
    const u8vec4 grid = unpack8(iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(iqs % 8) / 4]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid[iqs % 4] * (sign0 ? -1.0 : 1.0),
        grid[(iqs % 4) + 1] * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;

    const uint scale = (data_a[a_offset + ib].scales[ib32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qhshift = 2 * (ib8 % 4);
    const uint sign = data_a[a_offset + ib].qs[QUANT_K / 8 + ib8] >> (iqs % 8);

    const float db = 0.25 * (0.5 + scale);
    const u8vec4 grid = unpack8(iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(iqs % 8) / 4]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ3_XXS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint is = QUANT_K / 4 + 4 * ib32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[is / 2],
        data_a_packed16[a_offset + ib].qs[is / 2 + 1]));
    const float db = 0.5 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * (int(ib4 / 2) % 4), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq3xxs_grid[qs] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint is = QUANT_K / 4 + 4 * ib32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[is / 2],
        data_a_packed16[a_offset + ib].qs[is / 2 + 1]));
    const float db = 0.5 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * (int(ib4 / 2) % 4), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq3xxs_grid[qs]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ3_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint qs = data_a[a_offset + ib].qs[iqs / 4];
    const uint qh = data_a[a_offset + ib].qh[iqs / 32];
    const uint sign = data_a[a_offset + ib].signs[iqs / 8] >> (iqs % 8);
    const uint scale = data_a[a_offset + ib].scales[iqs / 64];
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    const float db = 1 + 2 * ((scale >> (4 * ((iqs / 32) & 1))) & 0xf);
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - ((iqs / 4) % 8))) & 256)] >> (8 * (iqs % 4));
    return db * vec2(
        int(grid & 0xFF) * (sign0 ? -1.0 : 1.0),
        int((grid >> 8) & 0xFF) * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint sign = data_a[a_offset + ib].signs[iqs / 8] >> (iqs % 8);
    const uint scale = data_a[a_offset + ib].scales[ib32 / 2];
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    const float db = 1 + 2 * ((scale >> (4 * (ib32 & 1))) & 0xf);
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - ib4 % 8)) & 256)] >> (8 * (iqs % 4));
    return db * vec4(
        int(grid & 0xFF) * (sign0 ? -1.0 : 1.0),
        int((grid >> 8) & 0xFF) * (sign1 ? -1.0 : 1.0),
        int((grid >> 16) & 0xFF) * (sign2 ? -1.0 : 1.0),
        int((grid >> 24) & 0xFF) * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ4_XS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint iq = 16 * ib32 + (iqs % 16);

    const uint sl = (data_a[a_offset + ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3;
    const uint qshift = (iqs & 16) >> 2;
    u8vec2 qs = u8vec2(data_a[a_offset + ib].qs[iq], data_a[a_offset + ib].qs[iq + 1]);
    qs = (qs >> qshift) & uint8_t(0xF);

    const float dl = float(int(sl | (sh << 4)) - 32);
    return dl * vec2(kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint iq = 16 * ib32 + (iqs % 16);

    const uint sl = (data_a[a_offset + ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3;
    const uint qshift = (iqs & 16) >> 2;
    const u8vec4 qs = unpack8((data_a_packed32[a_offset + ib].qs[iq/4] >> qshift) & 0x0F0F0F0F);

    const float dl = float(int(sl | (sh << 4)) - 32);
    return dl * vec4(
        kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y],
        kvalues_iq4nl[qs.z], kvalues_iq4nl[qs.w]);
}
#endif

#if defined(DATA_A_IQ4_NL)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(kvalues_iq4nl[vui & 0xF], kvalues_iq4nl[vui >> 4]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(kvalues_iq4nl[vui & 0xF], kvalues_iq4nl[(vui >> 4) & 0xF], kvalues_iq4nl[(vui >> 8) & 0xF], kvalues_iq4nl[vui >> 12]);
}
#endif

#if defined(DATA_A_MXFP4)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
#ifdef USE_OCP_FP4
    return vec2(unpackFloat2xfe2m1EXT(uint8_t(vui)));
#else
    return vec2(kvalues_mxfp4[vui & 0xF], kvalues_mxfp4[vui >> 4]) * 0.5;
#endif
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
#ifdef USE_OCP_FP4
    const uint16_t vui = uint16_t(uint(data_a[a_offset + ib].qs[iqs]) |
                                  uint(data_a[a_offset + ib].qs[iqs + 1]) << 8);
    return vec4(unpackFloat4xfe2m1EXT(vui));
#else
    vec2 v0 = dequantize(ib, iqs, a_offset);
    vec2 v1 = dequantize(ib, iqs + 1, a_offset);
    return vec4(v0.x, v0.y, v1.x, v1.y);
#endif
}
#endif

#if defined(DATA_A_NVFP4)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint sub = iqs >> 4;
    const float d = ue4m3_to_fp32(data_a[a_offset + ib].d[sub]);
    const uint j = iqs & 7;
    const uint shift = (iqs & 8) >> 1; // 0 or 4
#ifdef USE_OCP_FP4
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[(sub * 8u + j) / 2u]);
    return vec2(bitcastExtractfe2m1EXT(unpack8(vui).xy, shift)) * d;
#else
    const uint vui0 = uint(data_a[a_offset + ib].qs[sub * 8u + j]);
    const uint vui1 = uint(data_a[a_offset + ib].qs[sub * 8u + j + 1]);
    const uint qs0 = (vui0 >> shift) & 0xF;
    const uint qs1 = (vui1 >> shift) & 0xF;
    return vec2(float(kvalues_mxfp4[qs0]), float(kvalues_mxfp4[qs1])) * d * 0.5;
#endif
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
#ifdef USE_OCP_FP4
    const uint sub = iqs >> 4;
    const float d = ue4m3_to_fp32(data_a[a_offset + ib].d[sub]);
    const uint j = iqs & 7;
    const uint shift = (iqs & 8) >> 1; // 0 or 4
    const uint vui = data_a_packed32[a_offset + ib].qs[(sub * 8u + j) / 4u];
    return vec4(bitcastExtractfe2m1EXT(unpack8(vui), shift)) * d;
#else
    const vec2 v0 = dequantize(ib, iqs, a_offset);
    const vec2 v1 = dequantize(ib, iqs + 2u, a_offset);
    return vec4(v0.x, v0.y, v1.x, v1.y);
#endif
}
#endif

#if defined(DATA_A_F32) || defined(DATA_A_F16) || defined(DATA_A_BF16)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(0, 0);
}
#endif

#if defined(DATA_A_IQ1_M)
vec2 get_dm(uint ib, uint a_offset) {
    const uint16_t[4] scales = data_a[a_offset + ib].scales;
    const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
    const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);
    return vec2(d, 0);
}
#endif

#if defined(DATA_A_Q2_0) || defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0) || defined(DATA_A_IQ1_S) || defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S) || defined(DATA_A_IQ4_XS) || defined(DATA_A_IQ4_NL)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), 0);
}
#endif

#if defined(DATA_A_Q1_0)
vec2 get_dm(uint ib, uint a_offset) {
    const float d = float(data_a[a_offset + ib].d);
    return vec2(d, 0);
}
#endif

#if defined(DATA_A_MXFP4)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(e8m0_to_fp32(data_a[a_offset + ib].e), 0);
}
#endif

#if defined(DATA_A_NVFP4)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1.0, 0.0);
}
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
vec2 get_dm(uint ib, uint a_offset) {
    const vec2 dm = vec2(data_a_packed32[a_offset + ib].dm);
    return dm;
}
#endif

#if defined(DATA_A_Q2_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint qsi = (iqs / 64) * 32 + (iqs % 16) * 2; // 0,2,4..30
    const uint scalesi = iqs / 8;                      // 0..15
    const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

    const uvec2 qs = uvec2(data_a[a_offset + ib].qs[qsi], data_a[a_offset + ib].qs[qsi + 1]);
    const uint scales = data_a[a_offset + ib].scales[scalesi];
    const vec2 dm = vec2(data_a[a_offset + ib].dm);

    return dm.x * float(scales & 0xF) * vec2((qs >> qsshift) & 3) - dm.y * float(scales >> 4);
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_TQ1_0)
float tq1_0_val(uint ib, uint e, uint a_offset) {
    const uint bidx = tq1_0_byte_of(e);
    const uint qbyte = uint(bidx < 48u ? data_a[a_offset + ib].qs[bidx]
                                       : data_a[a_offset + ib].qh[bidx - 48u]);
    return float(tq1_0_trit(qbyte, tq1_0_digit_of(e))) - 1.0;
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(tq1_0_val(ib, iqs, a_offset), tq1_0_val(ib, iqs + 1u, a_offset));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), 0);
}
#endif

#if defined(DATA_A_TQ2_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    // elem e -> byte qs[(e/128)*32 + e%32], bits 2*((e%128)/32); w = q - 1 (d applied via get_dm)
    const uint qsi   = (iqs / 128) * 32 + (iqs % 32);  // iqs even -> qsi, qsi+1 in same group/level
    const uint shift = 2 * ((iqs % 128) / 32);

    const uvec2 qs = uvec2(data_a[a_offset + ib].qs[qsi], data_a[a_offset + ib].qs[qsi + 1]);
    return vec2((qs >> shift) & 3) - 1.0;
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), 0);
}
#endif

#if defined(DATA_A_Q3_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                     // 0,1
    const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
    const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
    const uint j = (iqs % 64) / 4;               // 0..3
    const uint is = iqs / 8;                     // 0..15
    const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
    const uint qsshift = halfsplit * 2;          // 0,2,4,6
    const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

    const int8_t us = int8_t(((data_a[a_offset + ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                          | (((data_a[a_offset + ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
    const float dl = float(data_a[a_offset + ib].d) * float(us - 32);

    return vec2(dl * float(int8_t((data_a[a_offset + ib].qs[qsi    ] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi    ] & m) != 0) ? 0 : 4)),
                dl * float(int8_t((data_a[a_offset + ib].qs[qsi + 1] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi + 1] & m) != 0) ? 0 : 4)));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q4_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

    const vec2 loadd = vec2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uint8_t sc = uint8_t((data_a[a_offset + ib].scales[scidx0] & 0xF) | ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uint8_t mbyte = uint8_t((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return vec2(fma(d, float((data_a[a_offset + ib].qs[qsi    ] >> (b * 4)) & 0xF), m),
                fma(d, float((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF), m));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q5_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
    const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

    const uint8_t hm = uint8_t(1 << (iqs / 16));

    const vec2 loadd = vec2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uint8_t sc    = uint8_t((data_a[a_offset + ib].scales[scidx0] & 0xF)                         | ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uint8_t mbyte = uint8_t(((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return vec2(fma(d, float((data_a[a_offset + ib].qs[qsi    ] >> (b * 4)) & 0xF) + float((data_a[a_offset + ib].qh[qhi    ] & hm) != 0 ? 16 : 0), m),
                fma(d, float((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF) + float((data_a[a_offset + ib].qh[qhi + 1] & hm) != 0 ? 16 : 0), m));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q6_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                    // 0,1
    const uint b = (iqs % 64) / 32;             // 0,1
    const uint is_b = (iqs % 16) / 8;           // 0,1
    const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
    const uint is = 8 * n + qhshift + is_b;     // 0..15
    const uint qsi = n * 64 + (iqs % 32) * 2;   // 0,2,4..126
    const uint qhi = n * 32 + (iqs % 16) * 2;   // 0,2,4..62

    const float dscale = float(data_a[a_offset + ib].d) * float(data_a[a_offset + ib].scales[is]);

    return vec2(dscale * float(int8_t(((data_a[a_offset + ib].ql[qsi    ] >> (b * 4)) & 0xF) | (((data_a[a_offset + ib].qh[qhi    ] >> qhshift) & 3) << 4)) - 32),
                dscale * float(int8_t(((data_a[a_offset + ib].ql[qsi + 1] >> (b * 4)) & 0xF) | (((data_a[a_offset + ib].qh[qhi + 1] >> qhshift) & 3) << 4)) - 32));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif


#if defined(DATA_A_IQ4_K)
const int kvalues_iq4_k[32] = int[32](
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18, -6,
    5, 17, 29, 42, 57, 73, 93, 117
);

vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint group = iqs / 32;
    const uint hf = (iqs / 16) & 1;
    const uint lane = iqs & 15;
    const uint high = (uint(data_a[a_offset + ib].scales_h[group / 2]) >> (4 * (group & 1))) & 15;
    const uint low = (uint(data_a[a_offset + ib].scales_l[group]) >> (4 * hf)) & 15;
    const int scale = int(low | (((high >> (2 * hf)) & 3) << 4)) - 32;
    const uint table = ((uint(data_a[a_offset + ib].extra) >> (2 * group + hf)) & 1) * 16;
    const uint code = (uint(data_a[a_offset + ib].qs[16 * group + lane]) >> (4 * hf)) & 15;
    const uint code1 = (uint(data_a[a_offset + ib].qs[16 * group + lane + 1]) >> (4 * hf)) & 15;
    const float dscale = float(data_a[a_offset + ib].d) * float(scale);

    return dscale * vec2(kvalues_iq4_k[table + code], kvalues_iq4_k[table + code1]);
}

vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_IQ2_K)
const int kvalues_iq2_k[8] = int[8](
    -31, -13, 1, 17, -26, -8, 6, 22
);

vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint hf = (iqs / 16) & 1;
    const uint lane = iqs & 15;
    const int scale = int((hf == 0 ? (uint(data_a[a_offset + ib].scales[ib32]) & 0xf) : (uint(data_a[a_offset + ib].scales[ib32]) >> 4))) - 8;
    const uint table = (((uint(data_a[a_offset + ib].extra) >> (2 * ib32 + hf)) & 1) != 0) ? 4 : 0;
    const uint qs_base = (ib32 < 4 ? 0 : 32) + (hf == 0 ? lane : 16 + lane);
    const uint shift = 2 * (ib32 % 4);
    const uint code0 = (uint(data_a[a_offset + ib].qs[qs_base]) >> shift) & 3;
    const uint code1 = (uint(data_a[a_offset + ib].qs[qs_base + 1]) >> shift) & 3;
    const float dscale = float(data_a[a_offset + ib].d) * float(scale);

    return dscale * vec2(kvalues_iq2_k[table + code0], kvalues_iq2_k[table + code1]);
}

vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_IQ3_K)
const int kvalues_iq3_k[16] = int[16](
    -63, -40, -23, -10, 1, 13, 28, 47,
    -59, -36, -19, -6, 5, 17, 32, 51
);

vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint hf = (iqs / 16) & 1;
    const uint lane = iqs & 15;
    const int raw = int(hf == 0 ? (uint(data_a[a_offset + ib].scales_l[ib32]) & 0xf) : (uint(data_a[a_offset + ib].scales_l[ib32]) >> 4));
    const int scale = (2 * raw + 1) * ((((uint(data_a[a_offset + ib].scales_h) >> (2 * ib32 + hf)) & 1) != 0) ? -1 : 1);
    const uint table = (((uint(data_a[a_offset + ib].extra) >> (2 * ib32 + hf)) & 1) != 0) ? 8 : 0;
    const uint qs_base = (ib32 < 4 ? 0 : 32) + (hf == 0 ? lane : 16 + lane);
    const uint shift = 2 * (ib32 % 4);
    const uint code0 = ((uint(data_a[a_offset + ib].qs[qs_base]) >> shift) & 3) | (((uint(data_a[a_offset + ib].qh[16 * hf + lane]) >> ib32) & 1) << 2);
    const uint code1 = ((uint(data_a[a_offset + ib].qs[qs_base + 1]) >> shift) & 3) | (((uint(data_a[a_offset + ib].qh[16 * hf + lane + 1]) >> ib32) & 1) << 2);
    const float dscale = float(data_a[a_offset + ib].d) * float(scale);

    return dscale * vec2(kvalues_iq3_k[table + code0], kvalues_iq3_k[table + code1]);
}

vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_IQ5_K)
const int kvalues_iq5_k[64] = int[64](
    -126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6, -1, 5, 11, 17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121,
    -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10, -4,  1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89, 99, 111, 123
);

vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib64 = iqs / 64;
    const uint sub16 = (iqs % 64) / 16;
    const uint lane = iqs % 16;
    const uint sh = uint(data_a[a_offset + ib].scales_h[ib64]);
    const uint sl = uint(data_a[a_offset + ib].scales_l[2*ib64 + sub16/2]);
    const int raw_l = int((sub16 % 2 == 0) ? (sl & 0xf) : (sl >> 4));
    const int raw_h = int((sub16 == 0) ? ((sh << 4) & 0x30) :
                          (sub16 == 1) ? ((sh << 2) & 0x30) :
                          (sub16 == 2) ? ((sh     ) & 0x30) :
                                         ((sh >> 2) & 0x30));
    const int scale = (raw_l | raw_h) - 32;
    const uint table = (((uint(data_a[a_offset + ib].extra) >> (4 * ib64 + sub16)) & 1) != 0) ? 32 : 0;
    const uint qs_base = 32 * ib64 + (sub16 % 2 == 0 ? lane : 16 + lane);
    const uint qh_base = (sub16 % 2 == 0 ? lane : 16 + lane);
    const uint shift = 2 * ib64;

    const uint low4_0 = (sub16 < 2) ? (uint(data_a[a_offset + ib].qs[qs_base]) & 0xf) : (uint(data_a[a_offset + ib].qs[qs_base]) >> 4);
    const uint high1_0 = (sub16 < 2) ? ((uint(data_a[a_offset + ib].qh[qh_base]) >> shift) & 1) : (((uint(data_a[a_offset + ib].qh[qh_base]) >> shift) & 2) >> 1);
    const uint code0 = low4_0 | (high1_0 << 4);

    const uint low4_1 = (sub16 < 2) ? (uint(data_a[a_offset + ib].qs[qs_base + 1]) & 0xf) : (uint(data_a[a_offset + ib].qs[qs_base + 1]) >> 4);
    const uint high1_1 = (sub16 < 2) ? ((uint(data_a[a_offset + ib].qh[qh_base + 1]) >> shift) & 1) : (((uint(data_a[a_offset + ib].qh[qh_base + 1]) >> shift) & 2) >> 1);
    const uint code1 = low4_1 | (high1_1 << 4);

    const float dscale = float(data_a[a_offset + ib].d) * float(scale);
    return dscale * vec2(kvalues_iq5_k[table + code0], kvalues_iq5_k[table + code1]);
}

vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_IQ6_K)
const int kvalues_iq6_k[128] = int[128](
    -127, -121, -115, -109, -104,  -98,  -93,  -88,  -84,  -79,  -74,  -70,  -66,  -62,  -58,  -54,
     -51,  -47,  -44,  -40,  -37,  -34,  -31,  -28,  -25,  -22,  -19,  -16,  -13,  -11,   -8,   -5,
      -2,    0,    3,    6,    9,   12,   14,   17,   20,   23,   27,   30,   33,   36,   40,   44,
      47,   51,   55,   59,   63,   68,   72,   77,   82,   87,   92,   98,  103,  109,  115,  121,
    -126, -120, -114, -108, -103,  -97,  -92,  -87,  -83,  -78,  -73,  -69,  -65,  -61,  -57,  -53,
     -50,  -46,  -43,  -39,  -36,  -33,  -30,  -27,  -24,  -21,  -18,  -15,  -12,  -10,   -7,   -4,
      -1,    1,    4,    7,   10,   13,   15,   18,   21,   24,   28,   31,   34,   37,   41,   45,
      48,   52,   56,   60,   64,   69,   73,   78,   83,   88,   93,   99,  104,  110,  116,  122
);

vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib64 = iqs / 64;
    const uint sub16 = (iqs % 64) / 16;
    const uint lane = iqs % 16;
    const int scale = int(data_a[a_offset + ib].scales[4 * ib64 + sub16]);
    const uint table = (((uint(data_a[a_offset + ib].extra) >> (4 * ib64 + sub16)) & 1) != 0) ? 64 : 0;
    const uint qs_base = 32 * ib64 + (sub16 % 2 == 0 ? lane : 16 + lane);
    const uint qh_base = 32 * (ib64 / 2) + (sub16 % 2 == 0 ? lane : 16 + lane);
    const uint shift = 4 * (ib64 % 2);

    const uint low4_0 = (sub16 < 2) ? (uint(data_a[a_offset + ib].qs[qs_base]) & 0xf) : (uint(data_a[a_offset + ib].qs[qs_base]) >> 4);
    const uint high2_0 = (sub16 < 2) ? (((uint(data_a[a_offset + ib].qh[qh_base]) >> shift) & 0x03) << 4) : (((uint(data_a[a_offset + ib].qh[qh_base]) >> shift) & 0x0c) << 2);
    const uint code0 = low4_0 | high2_0;

    const uint low4_1 = (sub16 < 2) ? (uint(data_a[a_offset + ib].qs[qs_base + 1]) & 0xf) : (uint(data_a[a_offset + ib].qs[qs_base + 1]) >> 4);
    const uint high2_1 = (sub16 < 2) ? (((uint(data_a[a_offset + ib].qh[qh_base + 1]) >> shift) & 0x03) << 4) : (((uint(data_a[a_offset + ib].qh[qh_base + 1]) >> shift) & 0x0c) << 2);
    const uint code1 = low4_1 | high2_1;

    const float dscale = float(data_a[a_offset + ib].d) * float(scale);
    return dscale * vec2(kvalues_iq6_k[table + code0], kvalues_iq6_k[table + code1]);
}

vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif
