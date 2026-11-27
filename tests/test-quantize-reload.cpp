#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...)                                      \
    do {                                                      \
        if (!(cond)) {                                        \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);       \
            printf(__VA_ARGS__);                              \
            printf("\n");                                     \
            n_fail++;                                         \
        }                                                     \
    } while (0)

static std::vector<float> to_f32(const ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, ggml_nbytes(t));
    } else {
        const auto * traits = ggml_get_type_traits(t->type);
        const int64_t ne0 = t->ne[0];
        const size_t row_size = ggml_row_size(t->type, ne0);
        for (int64_t r = 0; r < ggml_nrows(t); ++r) {
            traits->to_float((const char *) t->data + r * row_size, out.data() + r * ne0, ne0);
        }
    }
    return out;
}

static void test_ftype(const std::string & src_path, const std::string & dst_path, llama_ftype ftype, const char * name, ggml_type row_scaled, double max_nmse) {
    llama_model_quantize_params qp = llama_model_quantize_default_params();
    qp.ftype            = ftype;
    qp.allow_requantize = true;
    qp.nthread          = 2;

    CHECK(llama_model_quantize(src_path.c_str(), dst_path.c_str(), &qp) == 0, "%s: llama_model_quantize failed", name);
    if (!std::filesystem::exists(dst_path)) {
        return;
    }
    const size_t file_size = std::filesystem::file_size(dst_path);

    ggml_context * src_ctx = nullptr;
    gguf_context * src = gguf_init_from_file(src_path.c_str(), { false, &src_ctx });
    ggml_context * dst_ctx = nullptr;
    gguf_context * dst = gguf_init_from_file(dst_path.c_str(), { false, &dst_ctx });
    CHECK(src && dst, "%s: gguf_init_from_file failed", name);
    if (!src || !dst) {
        return;
    }

    CHECK(gguf_find_key(dst, "af_engn.format_version") >= 0, "%s: af_engn.format_version missing", name);

    const size_t align    = gguf_get_alignment(dst);
    const size_t data_off = gguf_get_data_offset(dst);
    const size_t meta     = gguf_get_meta_size(dst);
    const size_t padded   = (meta + align - 1) / align * align;
    CHECK(padded == data_off, "%s: data_off %zu != padded meta size %zu (header was resized after tensor data was placed)", name, data_off, padded);

    const int64_t n = gguf_get_n_tensors(dst);
    CHECK(n == gguf_get_n_tensors(src), "%s: tensor count mismatch", name);
    CHECK(gguf_get_tensor_offset(dst, 0) == 0, "%s: first tensor offset != 0", name);

    size_t expect_end = data_off;
    bool any_custom     = false;
    bool any_row_scaled = false;
    for (int64_t i = 0; i < n; ++i) {
        const char * tname = gguf_get_tensor_name(dst, i);
        const size_t off   = data_off + gguf_get_tensor_offset(dst, i);
        const size_t sz    = gguf_get_tensor_size(dst, i);
        CHECK(off % align == 0, "%s: tensor %s offset %zu unaligned", name, tname, off);
        CHECK(off + sz <= file_size, "%s: tensor %s [%zu, %zu) beyond EOF %zu", name, tname, off, off + sz, file_size);
        expect_end = off + (sz + align - 1) / align * align;

        ggml_tensor * dt = ggml_get_tensor(dst_ctx, tname);
        ggml_tensor * st = ggml_get_tensor(src_ctx, tname);
        CHECK(dt && st, "%s: tensor %s missing from ggml ctx", name, tname);
        if (!dt || !st) {
            continue;
        }
        if (dt->type != st->type) {
            any_custom = true;
        }
        if (dt->type == row_scaled) {
            any_row_scaled = true;
        }
        CHECK(ggml_validate_row_data_ne(dt->type, dt->data, ggml_nbytes(dt), dt->ne[0]), "%s: tensor %s failed row validation", name, tname);

        const std::vector<float> a = to_f32(st);
        const std::vector<float> b = to_f32(dt);
        double e = 0.0, r = 0.0;
        for (size_t k = 0; k < a.size(); ++k) {
            const double d = (double) a[k] - (double) b[k];
            e += d * d;
            r += (double) a[k] * (double) a[k];
        }
        const double nmse = r > 0 ? e / r : e;
        CHECK(std::isfinite(nmse) && nmse < max_nmse, "%s: tensor %s (%s) nmse %g >= %g", name, tname, ggml_type_name(dt->type), nmse, max_nmse);
        if (i == 0) {
            printf("%s: tensor 0 '%s' type %s nmse %g\n", name, tname, ggml_type_name(dt->type), nmse);
        }
    }
    CHECK(any_custom, "%s: no tensor was requantized", name);
    if (row_scaled != GGML_TYPE_COUNT) {
        CHECK(any_row_scaled, "%s: no tensor uses the row-scaled type %s", name, ggml_type_name(row_scaled));
    }
    CHECK(expect_end == file_size, "%s: file size %zu != expected end %zu", name, file_size, expect_end);

    gguf_free(src);
    gguf_free(dst);
    ggml_free(src_ctx);
    ggml_free(dst_ctx);

    llama_model_params mp = llama_model_default_params();
    mp.check_tensors = true;
    mp.n_gpu_layers  = 0;
    llama_model * model = llama_model_load_from_file(dst_path.c_str(), mp);
    CHECK(model != nullptr, "%s: llama_model_load_from_file failed", name);
    if (model) {
        llama_model_free(model);
    }
    printf("%s: OK (file %zu bytes, data_off %zu)\n", name, file_size, data_off);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <source.gguf> <workdir>\n", argv[0]);
        return 2;
    }
    const std::string src = argv[1];
    const std::string dir = argv[2];
    std::filesystem::create_directories(dir);

    llama_backend_init();

    struct { llama_ftype ftype; const char * name; ggml_type row_scaled; double max_nmse; } cases[] = {
        { LLAMA_FTYPE_MOSTLY_IQ5_K,   "IQ5_K",   GGML_TYPE_COUNT , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ6_K,   "IQ6_K",   GGML_TYPE_COUNT , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ4_K,   "IQ4_K",   GGML_TYPE_COUNT , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ3_K,   "IQ3_K",   GGML_TYPE_COUNT , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ2_K,   "IQ2_K",   GGML_TYPE_COUNT , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ5_KS,  "IQ5_KS",  GGML_TYPE_IQ5_KS  , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ4_KS,  "IQ4_KS",  GGML_TYPE_IQ4_KS  , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ4_KSS, "IQ4_KSS", GGML_TYPE_IQ4_KSS , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ3_KS,  "IQ3_KS",  GGML_TYPE_IQ3_KS  , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ2_KS,  "IQ2_KS",  GGML_TYPE_IQ2_KS  , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ4_KT,  "IQ4_KT",  GGML_TYPE_IQ4_KT  , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ3_KT,  "IQ3_KT",  GGML_TYPE_IQ3_KT  , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ2_KT,  "IQ2_KT",  GGML_TYPE_IQ2_KT  , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ1_KT,  "IQ1_KT",  GGML_TYPE_IQ1_KT  , 0.4 },
        { LLAMA_FTYPE_MOSTLY_IQ2_KL,  "IQ2_KL",  GGML_TYPE_IQ2_KL  , 0.2 },
        { LLAMA_FTYPE_MOSTLY_IQ1_BN,  "IQ1_BN",  GGML_TYPE_IQ1_BN  , 0.9 },
    { LLAMA_FTYPE_MOSTLY_IQ2_BN,  "IQ2_BN",  GGML_TYPE_IQ2_BN  , 0.9 },
    { LLAMA_FTYPE_MOSTLY_Q6_0,  "Q6_0",  GGML_TYPE_COUNT  , 0.05 },
    { LLAMA_FTYPE_MOSTLY_PQ2_0,   "PQ2_0",   GGML_TYPE_COUNT , 0.9 },
    };
    for (const auto & c : cases) {
        test_ftype(src, dir + "/out-" + c.name + ".gguf", c.ftype, c.name, c.row_scaled, c.max_nmse);
    }

    llama_backend_free();
    if (n_fail) {
        printf("test-quantize-reload: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-quantize-reload: OK\n");
    return 0;
}
