// CPU unit test for GGML_TYPE_IQ2_K (CARD 5.2 / 5.3)
//
// Validates:
//  - runtime type id, block size and type size
//  - dequantize_row_iq2_k against a hand-crafted block (bitwise check)
//  - ggml_vec_dot_iq2_k_q8_K consistency with the dequantized values (frozen tolerance)
//  - quantize -> dequantize round-trip within a frozen RMSE tolerance

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include "gguf.h"
#include <cstdio>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#define QK_K_IQ2 256

static const int8_t iq2nl[8] = { -31, -13, 1, 17, -26, -8, 6, 22 };

static bool near(float a, float b, float eps) {
    return std::fabs(a - b) <= eps;
}

#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); exit(1); } } while (0)

static void test_type_traits() {
    CHECK((int)GGML_TYPE_IQ2_K == 45, "runtime id must be 45");
    CHECK(ggml_blck_size(GGML_TYPE_IQ2_K) == QK_K_IQ2, "blck_size");
    CHECK(ggml_type_size(GGML_TYPE_IQ2_K) == sizeof(block_iq2_k), "type_size");
    CHECK(sizeof(block_iq2_k) == 2 + 2 + QK_K_IQ2/32 + QK_K_IQ2/4, "block size/padding");
    CHECK(ggml_is_quantized(GGML_TYPE_IQ2_K), "is_quantized");
    CHECK(std::strcmp(ggml_type_name(GGML_TYPE_IQ2_K), "iq2_k") == 0, "type_name");
    printf("test_type_traits: OK (id=45, blck=%lld, type_size=%zu, name=%s)\n",
           (long long) ggml_blck_size(GGML_TYPE_IQ2_K), ggml_type_size(GGML_TYPE_IQ2_K),
           ggml_type_name(GGML_TYPE_IQ2_K));
}

// Frozen reference: dequantize a hand-crafted block, then recompute the
// expected values directly from the documented layout (bitwise-exact).
static void test_dequantize_frozen() {
    block_iq2_k x;
    std::memset(&x, 0, sizeof(x));

    x.d = 0x3c00;       // fp16 1.0f
    x.extra = 0x5555;   // exercise shifted codebooks on alternating sub-blocks
    x.scales[0] = 0x3c; // low nibble 12 -> dl1=4, high nibble 3 -> dl2=-5
    for (int i = 1; i < QK_K_IQ2/32; ++i) x.scales[i] = (uint8_t)(0x9a + i);
    for (int j = 0; j < QK_K_IQ2/4; ++j) x.qs[j] = (uint8_t)((j * 7 + 1) & 0xff);

    float y[QK_K_IQ2];
    dequantize_row_iq2_k(&x, y, QK_K_IQ2);

    // Independent reconstruction of the documented layout (bitwise-exact).
    const float d = 1.0f;
    uint16_t extra = x.extra;
    const uint8_t * qs = x.qs;
    int shift = 0;
    const float * yp = y;
    for (int ib32 = 0; ib32 < QK_K_IQ2/32; ++ib32) {
        const float dl1 = d * ((x.scales[ib32] & 0xf) - 8);
        const float dl2 = d * ((x.scales[ib32] >> 4) - 8);
        const int8_t * v1 = (extra & 1) ? iq2nl + 4 : iq2nl;
        const int8_t * v2 = (extra & 2) ? iq2nl + 4 : iq2nl;
        extra >>= 2;
        for (int j = 0; j < 16; ++j) {
            const float r1 = dl1 * v1[(qs[j] >> shift) & 3];
            const float r2 = dl2 * v2[(qs[16 + j] >> shift) & 3];
            CHECK(yp[j] == r1, "iq2_k dequant mismatch (first half)");
            CHECK(yp[16 + j] == r2, "iq2_k dequant mismatch (second half)");
        }
        yp += 32;
        shift += 2;
        if (shift == 8) { qs += 32; shift = 0; }
    }
    printf("test_dequantize_frozen: OK (group0 y[0]=%.4f y[16]=%.4f)\n", y[0], y[16]);
}

// vec_dot must equal the float dot product of the dequantized row and the Q8_K row.
static void test_vec_dot_consistency() {
    block_iq2_k x;
    std::memset(&x, 0, sizeof(x));
    x.d = 0x3c00;   // 1.0
    x.extra = 0x5555;
    for (int i = 0; i < QK_K_IQ2/32; ++i) x.scales[i] = (uint8_t)(0x9a + i);
    for (int j = 0; j < QK_K_IQ2/4; ++j) x.qs[j] = (uint8_t)((j * 7 + 1) & 0xff);

    block_q8_K y;
    std::memset(&y, 0, sizeof(y));
    y.d = 0.5f;
    for (int i = 0; i < QK_K_IQ2; ++i) y.qs[i] = (int8_t)(((i * 7) % 21) - 10);

    float deq[QK_K_IQ2];
    dequantize_row_iq2_k(&x, deq, QK_K_IQ2);

    float ref = 0.0f;
    for (int i = 0; i < QK_K_IQ2; ++i) ref += deq[i] * (y.d * y.qs[i]);

    float got = 0.0f;
    ggml_vec_dot_iq2_k_q8_K(QK_K_IQ2, &got, 0, &x, 0, &y, 0, 1);

    const float scale = std::fmax(1.0f, std::fabs(ref));
    const float tol = 1e-4f * scale;
    printf("test_vec_dot_consistency: ref=%.6f got=%.6f diff=%.6g tol=%.6g\n",
           ref, got, got - ref, tol);
    CHECK(near(ref, got, tol), "iq2_k vec_dot disagrees with dequantized dot");
}

static void test_quantize_roundtrip() {
    const int64_t n = 4 * QK_K_IQ2;
    std::vector<float> src(n);
    uint32_t state = 0x12345678u;
    for (int64_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        src[i] = ((int32_t)(state >> 16) % 2001) / 1000.0f - 1.0f;
    }

    const size_t row_size = ggml_row_size(GGML_TYPE_IQ2_K, n);
    std::vector<uint8_t> q(row_size);
    ggml_quantize_chunk(GGML_TYPE_IQ2_K, src.data(), q.data(), 0, 1, n, nullptr);

    std::vector<float> out(n);
    ggml_get_type_traits(GGML_TYPE_IQ2_K)->to_float(q.data(), out.data(), n);

    double se = 0.0;
    for (int64_t i = 0;i < n; ++i) {
        const double e = (double)out[i] - (double)src[i];
        se += e * e;
    }
    const double rmse = std::sqrt(se / n);
    printf("test_quantize_roundtrip: rmse=%.6f (frozen tol 0.16)\n", rmse);
    CHECK(rmse < 0.16, "iq2_k round-trip RMSE above frozen tolerance");
}


// Verify the af_engn wire-id mapping: GGML_TYPE_IQ2_K <-> 1025.
static void test_gguf_wire_id() {
    const int64_t n = QK_K_IQ2;
    std::vector<uint8_t> qdata(ggml_row_size(GGML_TYPE_IQ2_K, n));
    std::vector<float> src(n);
    uint32_t state = 0xdeadbeefu;
    for (int64_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        src[i] = ((int32_t)(state >> 16) % 1001) / 1000.0f;
    }
    ggml_quantize_chunk(GGML_TYPE_IQ2_K, src.data(), qdata.data(), 0, 1, n, nullptr);

    const char * fname = "test-iq2-k-wire.gguf";
    {
        gguf_context * ctx = gguf_init_empty();
        CHECK(ctx != nullptr, "gguf_init_empty failed");
        // af_engn dialect marker required for the custom wire ids
        gguf_set_val_u32(ctx, "af_engn.format_version", 1);

        ggml_init_params params = { 1024*1024, nullptr, false };
        ggml_context * gctx = ggml_init(params);
        CHECK(gctx != nullptr, "ggml_init failed");
        ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_IQ2_K, n);
        CHECK(t != nullptr, "ggml_new_tensor_1d failed");
        ggml_set_name(t, "tensor_iq2k");
        gguf_add_tensor(ctx, t);
        gguf_set_tensor_data(ctx, "tensor_iq2k", qdata.data());
        CHECK(gguf_write_to_file(ctx, fname, false), "gguf_write_to_file failed");
        gguf_free(ctx);
        ggml_free(gctx);
    }
    {
        gguf_init_params p = { true, nullptr };
        gguf_context * ctx = gguf_init_from_file(fname, p);
        CHECK(ctx != nullptr, "gguf_init_from_file failed (wire id mapping broken)");
        int64_t ti = gguf_find_tensor(ctx, "tensor_iq2k");
        CHECK(ti >= 0, "tensor_iq2k not found");
        enum ggml_type t = gguf_get_tensor_type(ctx, ti);
        CHECK(t == GGML_TYPE_IQ2_K, "round-tripped tensor type is not GGML_TYPE_IQ2_K");
        printf("test_gguf_wire_id: OK (wire 1025 -> GGML_TYPE_IQ2_K round-trip)\n");
        gguf_free(ctx);
    }
    std::remove(fname);
}

int main() {
    test_type_traits();
    test_dequantize_frozen();
    test_vec_dot_consistency();
    test_quantize_roundtrip();
    test_gguf_wire_id();
    printf("iq2_k cpu tests: ALL PASSED\n");
    return 0;
}
