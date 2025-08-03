// ggml-gemmini-util.h
#ifndef DEBUG
#define DEBUG 0
#endif

#include "ggml-impl.h"
#include "ggml-gemmini.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <map>
#include <set>
#include <cstring>

#ifndef PRINT_TILE
#define PRINT_TILE 0
#endif

#if DEBUG 
    #define DBG(fmt, ...) \
        fprintf(stderr, "[%s:%d] %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
    #define DBG(fmt, ...)  ((void)0)
#endif


struct ggml_backend_gemmini_context
{
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
    std::map<ggml_tensor *, ggml_tensor *> bias_map;
    struct ggml_context *tmp_ctx = nullptr;
    void *arena = nullptr;
    bool tmp_ctx_initialized = false;

#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
};

namespace zerogod
{   
    constexpr size_t GEMMINI_ALIGN = 16; // 16-byte align

    static inline size_t align_up(size_t val, size_t align)
    {
        return (val + align - 1) / align * align;
    }
    
    static void ggml_calc_tmp_ctx_size(ggml_cgraph *cgraph,
                                       ggml_backend_gemmini_context *ctx,
                                       size_t &peak_bytes,
                                       size_t &peak_meta)
    {
        peak_bytes = peak_meta = 0;

        enum class role_t { ACC,   // 출력 C  (int32)
                            SRC,   // A,B     (int8)
                            BIAS }; // D      (int32)


        auto calc_one = [&](ggml_tensor *t, role_t role, bool transpose = false, int row_pad = -1) -> std::pair<size_t, size_t> {
            size_t meta = ggml_tensor_overhead();
            size_t bytes = 0;

            switch (role) {
            case role_t::ACC: {
                size_t row_e = zerogod::align_up((size_t)t->ne[transpose ? 0 : 1], 16);
                bytes = zerogod::align_up(row_e * sizeof(GGML_TYPE_I32), GEMMINI_ALIGN) * (size_t)t->ne[transpose ? 1 : 0];
                break;
            }
            case role_t::SRC: {
                const size_t cols_orig = transpose ? t->ne[1] : t->ne[0];
                const size_t rows_orig = transpose ? t->ne[0] : t->ne[1];
                const size_t cols_e = zerogod::align_up(cols_orig, 16);
                const size_t rows_e = zerogod::align_up(rows_orig, 16);

                size_t row_b = cols_e * ggml_type_size(GGML_TYPE_I8);
                row_b = align_up(row_b, GEMMINI_ALIGN);
                bytes = row_b * rows_e;
                break;
            }
            case role_t::BIAS: {
                size_t row_b = zerogod::align_up((size_t)t->ne[0] * ggml_type_size(GGML_TYPE_I32), 16);
                bytes = zerogod::align_up(row_b, GEMMINI_ALIGN) * (size_t)t->ne[1];
                break;
            }

            }
            return {bytes, meta};
        };

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            size_t data_sum = 0, meta_sum = 0;

            auto *node = cgraph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT) continue;

            // B 텐서의 실제 열 개수 J, 그리고 16단위로 패딩된 J_pad
            const int J = node->src[1]->ne[1];
            const int J_pad = align_up(J, 16);

            // A: transpose = false, row_pad = -1
            auto [bA, mA] = calc_one(node->src[0], role_t::SRC, false, -1);
            data_sum += bA;
            meta_sum += mA;

            // B: transpose = true, row_pad = J_pad
            auto [bB, mB] = calc_one(node->src[1], role_t::SRC, true, J_pad);
            data_sum += bB;
            meta_sum += mB;

            // bias (optional)
            if (auto it = ctx->bias_map.find(node); it != ctx->bias_map.end()) {
                auto [b,m] = calc_one(it->second, role_t::BIAS);
                data_sum += b;
                meta_sum += m;
            }
            // C
            auto [bC,mC] = calc_one(node, role_t::ACC);
            data_sum += bC;
            meta_sum += mC;

            peak_bytes = std::max(peak_bytes, data_sum);
            peak_meta  = std::max(peak_meta , meta_sum );
        }

        // 여유 16 KiB 및 행 정렬
        peak_bytes = zerogod::align_up(peak_bytes + 16*1024, GEMMINI_ALIGN);
    }

    template <typename INT_T> // 사용 X
    static ggml_tensor *ggml_cast_tensor(ggml_context *ctx,
                                         const ggml_tensor *src,
                                         bool fill_from_src,
                                         const char *suffix,
                                         bool   swap_dims = false,
                                         int row_pad = -1)
    {
        // 1) 출력 차원 결정
        const int cols_orig = swap_dims ? src->ne[1] : src->ne[0];   // 실제 열
        const int cols_pad  = align_up(cols_orig, 16);      // Gemmini 타일(16) 단위

        const int rows_orig = swap_dims ? src->ne[0] : src->ne[1];   // 행
        const int rows_pad = row_pad > 0 ? row_pad : align_up(rows_orig, 16);

        auto type = std::is_same<INT_T,int8_t>::value ? GGML_TYPE_I8 : GGML_TYPE_I32;

        // 2) 패딩 계산
        constexpr size_t ELEM = sizeof(INT_T);
        const size_t row_bytes_orig = cols_orig * ELEM;
        const size_t row_bytes_pad = cols_pad * ELEM;

        const size_t padded = align_up(row_bytes_pad, GEMMINI_ALIGN);
        const size_t stride_e = padded / ELEM;

        // 3) 텐서 생성
        ggml_tensor *q = ggml_new_tensor_2d(ctx, type, stride_e, rows_pad);
        snprintf(q->name, sizeof(q->name), "%s%s", src->name, suffix);


        uint8_t *dst_row = (uint8_t *)q->data;
        const float *src_f = (const float *)src->data;

        for (int r = 0; r < rows_pad; ++r)
        {
            INT_T *dst_elem = (INT_T *)dst_row;

            if (r < rows_orig && fill_from_src) {
                if (!swap_dims) {
                    // 1-to-1 복사
                    for (int c = 0; c < cols_orig; ++c)
                        dst_elem[c] = (INT_T)src_f[r * cols_orig + c];
                } else {
                    // 전치 복사  (src : K×J, dst : J×K)
                    for (int c = 0; c < cols_orig; ++c)
                        dst_elem[c] = (INT_T)src_f[c * rows_orig + r];
                }

                memset(dst_elem + cols_orig, 0, row_bytes_pad - row_bytes_orig);
            } else memset(dst_elem, 0, row_bytes_pad);


            if (padded > row_bytes_pad)
                memset(dst_row + row_bytes_pad, 0, padded - row_bytes_pad);

            dst_row += padded;
        }


        // 4) stride(nb) 재계산
        q->nb[0] = ELEM;
        q->nb[1] = padded;
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            q->nb[d] = q->nb[d - 1] * q->ne[d - 1];
        }

        DBG1("cast: %-24s -> (%dx%d) type=%s nb1=%zu\n", q->name, rows_pad, cols_pad,
             std::is_same<INT_T, int8_t>::value ? "i8" : "i32", q->nb[1]);

        return q;
    }
}

