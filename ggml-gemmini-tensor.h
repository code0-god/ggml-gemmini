// ggml-gemmini-tensor.h
#ifndef __GGML_GEMMINI_TENSOR_H__
#define __GGML_GEMMINI_TENSOR_H__

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cstdlib>
#include <cstring>

#include "ggml.h"
#include "ggml-gemmini-util.h"

namespace zerogod
{
    template <typename T>
    class ggml_gemmini_tensor
    {
        static_assert(std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value,
                      "T must be int8_t or int32_t");

    public:
        ggml_gemmini_tensor(ggml_context *ctx,
                            const ggml_tensor *src,
                            const char *suffix = "_cast",
                            bool acc = false,
                            bool transpose = false);

        ~ggml_gemmini_tensor();

        // 이동 전용 구현
        ggml_gemmini_tensor(ggml_gemmini_tensor &&) noexcept;            // 이동 생성자
        ggml_gemmini_tensor &operator=(ggml_gemmini_tensor &&) noexcept; // 이동 대입

        ggml_gemmini_tensor(const ggml_gemmini_tensor &) = delete;            // 복사 생성자 금지
        ggml_gemmini_tensor &operator=(const ggml_gemmini_tensor &) = delete; // 복사 대입 금지

        // Gemmini 커널용 데이터 버퍼
        void *get() noexcept { return data_; }
        const void *get() const noexcept { return data_; }

    private:
        void ggml_gemmini_cast(const ggml_tensor *src, size_t dst_cols, bool transpose) const; // data casting
        void update_stride();                                             // stride 재계산
        void free_buffer();

        ggml_tensor *tensor_ = nullptr; // 변환된 텐서
        void *data_ = nullptr;          // casting & align된 data 버퍼
        size_t buf_bytes_ = 0;          // 할당된 바이트 수
    };

    // explicit instantiation : 지원 타입 한정
    extern template class ggml_gemmini_tensor<int8_t>;
    extern template class ggml_gemmini_tensor<int32_t>;
}

#endif // __GGML_GEMMINI_TENSOR_H__