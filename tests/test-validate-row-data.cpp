#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
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

static const ggml_type ROW_SCALED[] = {
    GGML_TYPE_IQ2_KS, GGML_TYPE_IQ3_KS, GGML_TYPE_IQ4_KS,  GGML_TYPE_IQ4_KSS, GGML_TYPE_IQ5_KS,
    GGML_TYPE_IQ1_KT, GGML_TYPE_IQ2_KT, GGML_TYPE_IQ3_KT,  GGML_TYPE_IQ4_KT,  GGML_TYPE_IQ2_KL,
    GGML_TYPE_IQ1_BN,
    GGML_TYPE_IQ2_BN,
};

static const ggml_type ROW_PLAIN[] = {
    GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ2_K, GGML_TYPE_IQ3_K, GGML_TYPE_IQ4_K, GGML_TYPE_IQ5_K, GGML_TYPE_IQ6_K,
    GGML_TYPE_Q6_0,
    GGML_TYPE_PQ2_0,
    GGML_TYPE_PTQ1_0,
};

static void set_row_scale(std::vector<uint8_t> & buf, size_t row, size_t row_size, int64_t meta, uint32_t bits) {
    uint8_t * p = buf.data() + row * row_size;
    if (meta == 4) {
        memcpy(p, &bits, 4);
    } else {
        uint16_t h = (uint16_t) bits;
        memcpy(p, &h, 2);
    }
}

static void test_type(ggml_type type, int64_t ne0, int64_t nrows) {
    const char * name = ggml_type_name(type);
    const int64_t meta = ggml_row_meta_size(type);
    const size_t row_size = ggml_row_size(type, ne0);

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> src(ne0 * nrows);
    for (auto & v : src) {
        v = dist(rng);
    }
    std::vector<uint8_t> dst(row_size * nrows);
    const size_t written = ggml_quantize_chunk(type, src.data(), dst.data(), 0, nrows, ne0, nullptr);
    CHECK(written == row_size * nrows, "%s ne0=%lld nrows=%lld: written %zu != %zu", name, (long long) ne0, (long long) nrows, written, row_size * nrows);

    CHECK(ggml_validate_row_data_ne(type, dst.data(), written, ne0), "%s ne0=%lld nrows=%lld: valid data rejected", name, (long long) ne0, (long long) nrows);

    if (meta != 0) {
        CHECK(written % ggml_type_size(type) != 0 || row_size % ggml_type_size(type) != 0 || ggml_blck_size(type) == 1,
              "%s: row size unexpectedly a multiple of type size", name);
        CHECK(!ggml_validate_row_data(type, dst.data(), written), "%s: row-width-less validation must refuse row-scaled types", name);
    }

    std::vector<uint8_t> big(dst);
    big.push_back(0);
    CHECK(!ggml_validate_row_data_ne(type, big.data(), written + 1, ne0), "%s: size+1 accepted", name);
    CHECK(!ggml_validate_row_data_ne(type, dst.data(), written - 1, ne0), "%s: size-1 accepted", name);
    CHECK(!ggml_validate_row_data_ne(type, dst.data(), 0, ne0), "%s: size 0 accepted", name);
    CHECK(!ggml_validate_row_data_ne(type, dst.data(), written, ne0 + 1), "%s: ne0+1 accepted", name);
    CHECK(!ggml_validate_row_data_ne(type, dst.data(), written, 0), "%s: ne0=0 accepted", name);

    if (meta == 2 || meta == 4) {
        const uint32_t nan_bits = meta == 4 ? 0x7fc00000u : 0x7e00u;
        const uint32_t inf_bits = meta == 4 ? 0x7f800000u : 0x7c00u;
        for (int64_t r : {int64_t(0), nrows - 1}) {
            std::vector<uint8_t> bad(dst);
            set_row_scale(bad, r, row_size, meta, nan_bits);
            CHECK(!ggml_validate_row_data_ne(type, bad.data(), written, ne0), "%s row %lld: NaN scale accepted", name, (long long) r);
            set_row_scale(bad, r, row_size, meta, inf_bits);
            CHECK(!ggml_validate_row_data_ne(type, bad.data(), written, ne0), "%s row %lld: Inf scale accepted", name, (long long) r);
        }
    }
}

int main() {
    for (ggml_type type : ROW_SCALED) {
        CHECK(ggml_row_meta_size(type) != 0, "%s: expected per-row metadata", ggml_type_name(type));
        for (int64_t nrows : {1, 7}) {
            test_type(type, 2560, nrows);
        }
        const bool tails = type == GGML_TYPE_IQ1_KT || type == GGML_TYPE_IQ2_KT || type == GGML_TYPE_IQ3_KT || type == GGML_TYPE_IQ4_KT;
        if (tails) {
            test_type(type, 2560 + 32, 7);
        }
    }

    for (ggml_type type : ROW_PLAIN) {
        CHECK(ggml_row_meta_size(type) == 0, "%s: unexpected per-row metadata", ggml_type_name(type));
        for (int64_t nrows : {1, 7}) {
            test_type(type, 2560, nrows);
        }
    }

    // the exact failing case from the bug report: 7 rows of IQ5_KS at ne0 = 2560
    CHECK(ggml_row_size(GGML_TYPE_IQ5_KS, 2560) == 1684, "IQ5_KS row size at 2560: %zu", ggml_row_size(GGML_TYPE_IQ5_KS, 2560));
    CHECK(7 * ggml_row_size(GGML_TYPE_IQ5_KS, 2560) == 11788, "IQ5_KS 7-row chunk size");

    if (n_fail) {
        printf("test-validate-row-data: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-validate-row-data: OK\n");
    return 0;
}
