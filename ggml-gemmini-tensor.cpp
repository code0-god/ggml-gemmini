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
        
        DBG("generate ggml_gemmini_tensor: %s, transpose=%d\n", src->name, transpose);

        /* 1. ____________________원본 행/열____________________ 
              ggml 네이티브: ne[0] = columns(X), ne[1] = rows(Y) */
        const int src_cols = transpose ? src->ne[1] : src->ne[0];
        const int src_rows = transpose ? src->ne[0] : src->ne[1]; 
        auto type = ggml_type_of<T>();

        /* 2. _____16-byte row-stride 정렬을 위한 colum 패딩_____ */
        const size_t elem_size = ggml_type_size(type);
        const size_t align_elems = GEMMINI_ALIGN / elem_size;
        const int padded_cols = align_up(src_cols, align_elems);
        
        /* 3. ___________________tensor 생성___________________ */
        tensor_ = ggml_new_tensor_2d(ctx, type, padded_cols, src_rows);
        snprintf(tensor_->name, sizeof(tensor_->name), "%s%s", src->name, suffix);

        DBG("generated tensor: cols=%d, rows=%d\n", tensor_->ne[0], tensor_->ne[1]);

        /* 4. ______________data 복제 및 casting_______________ */
        ggml_gemmini_cast(src_rows, padded_cols, transpose);

        /* 5. _________________stride 업데이트__________________ */
        update_stride();
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

    template <typename T>
    void ggml_gemmini_tensor<T>::ggml_gemmini_cast(size_t rows, size_t cols, bool transpose)
    {
        // TODO:
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
