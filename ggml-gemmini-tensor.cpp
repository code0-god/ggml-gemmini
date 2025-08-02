// ggml-gemmini-tensor.cpp
#include "ggml-gemmini-tensor.h"

namespace zerogod
{
    template <typename T>
    static inline ggml_type ggml_type_of()
    {
        return std::is_same<T, int8_t>::value ? GGML_TYPE_I8
                                              : GGML_TYPE_I32;
    }

    // 생성자
    template <typename T>
    ggml_gemmini_tensor<T>::ggml_gemmini_tensor(ggml_context *ctx,
                                                const ggml_tensor *src,
                                                const char *suffix,
                                                bool transpose)
    {
        // TODO: 16-byte align & integer casting
    }

    // 소멸자 & 버퍼 해제
    template <typename T>
    ggml_gemmini_tensor<T>::~ggml_gemmini_tensor() { free_buffer(); }

    template <typename T>
    void ggml_gemmini_tensor<T>::free_buffer()
    {
        if (data_)
        {
            std::free(data_);
            data_ = nullptr;
        }
        tensor_ = nullptr;
        buf_bytes_ = 0;
    }

    // 이동 생성자 & 이동 대입 연산자 오버라이딩
    // other: 기존 객체
    template <typename T>
    ggml_gemmini_tensor<T>::ggml_gemmini_tensor(ggml_gemmini_tensor &&other) noexcept
        : tensor_(other.tensor_), data_(other.data_), buf_bytes_(other.buf_bytes_)
    {
        other.tensor_ = nullptr;
        other.data_ = nullptr;
        other.buf_bytes_ = 0;
    }

    template <typename T>
    ggml_gemmini_tensor<T> &
    ggml_gemmini_tensor<T>::operator=(ggml_gemmini_tensor &&other) noexcept
    {
        if (this != &other)
        {
            free_buffer();
            tensor_ = other.tensor_;
            data_ = other.data_;
            buf_bytes_ = other.buf_bytes_;
            other.tensor_ = nullptr;
            other.data_ = nullptr;
            other.buf_bytes_ = 0;
        }
        return *this;
    }

    template <typename T>
    void ggml_gemmini_tensor<T>::update_stride()
    {
        for (int d = 2; d < GGML_MAX_DIMS; ++d)
            tensor_->nb[d] = tensor_->nb[d - 1] * tensor_->ne[d - 1];
    }

    // explicit instantiation : 지원 타입 한정
    template class ggml_gemmini_tensor<int8_t>;
    template class ggml_gemmini_tensor<int32_t>;

} // namespace zerogod
