// ggml-gemmini-util.h
#include "ggml-impl.h"
#include "ggml-gemmini.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <map>
#include <set>
#include <cstring>

#include "include/gemmini.h"

#ifndef PRINT_TILE
#define PRINT_TILE 0
#endif
#define DBG(fmt, ...) \
    fprintf(stderr, "[%s:%d] %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)

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
    static inline size_t align_up(size_t val, size_t align)
    {
        return (val + align - 1) / align * align;
    }
    
    static void ggml_calc_tmp_ctx_size_impl(ggml_tensor *t,
        size_t elem_size,
        std::set<ggml_tensor *> &qset,
        size_t &total_bytes,
        size_t &total_meta)
        {
            if (qset.insert(t).second)
        {
            // 원본 tensor 실제 byte 수
            const size_t row_bytes = t->ne[0] * elem_size;
            
            // Gemmini용으로 16B align 시 필요한 총 byte 수
            const size_t padded = align_up(row_bytes, 16); // 16B 정렬
            const size_t bytes = padded * t->ne[1];
            
            total_bytes += bytes;
            total_meta += ggml_tensor_overhead();
        }
    }
    
    static void ggml_calc_tmp_ctx_size(ggml_cgraph *cgraph, ggml_backend_gemmini_context *ctx, size_t elem_size, std::set<ggml_tensor *> &qset, size_t &total_bytes, size_t &total_meta)
    {
        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            auto *node = cgraph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT)
                continue;

            // A, B → int8
            for (auto *t : {node->src[0], node->src[1]})
                ggml_calc_tmp_ctx_size_impl(t, sizeof(int8_t), qset, total_bytes, total_meta);

            // bias → int32 (optional)
            auto it = ctx->bias_map.find(node);
            if (it != ctx->bias_map.end())
            {
                auto *bias = it->second;
                ggml_calc_tmp_ctx_size_impl(bias, sizeof(int32_t), qset, total_bytes, total_meta);
            }

            // C(acc) → int32
            ggml_calc_tmp_ctx_size_impl(node, sizeof(int32_t), qset, total_bytes, total_meta);
        }
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

        const size_t padded = align_up(row_bytes_pad, GEMMINI_ROW_ALIGN);
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

