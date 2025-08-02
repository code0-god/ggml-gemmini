#include "ggml-gemmini-tensor.h"

static void ggml_backend_gemmini_mul_mat(
                                         ggml_backend_gemmini_context *ctx,
                                         struct ggml_tensor *dst, // FP32 output (I×J)
                                         struct ggml_tensor *bias) // optional FP32 bias (→int32)
{
    DBG("[Gemmini] mul_mat call\n");

    // 0. 원본 FP32 입력 텐서
    const auto *src0 = dst->src[0];         // A:  I×K
    const auto *src1 = dst->src[1];         // B:  K×J

    const size_t I = src0->ne[1];   // rows  (A 의 두 번째 차원)
    const size_t J = src1->ne[1];   // cols  (B 의 두 번째 차원)
    const size_t K = src0->ne[0];

    const size_t J_pad = zerogod::align_up(J, 16);

    DBG("mul_mat entry: I=%zu, J=%zu, J_pad=%zu, K=%zu\n", I, J, J_pad, K);

    // 1. int8 캐스팅 (패딩 16 B)
    auto *tA = zerogod::ggml_cast_tensor<int8_t>(ctx->tmp_ctx, src0, true, ".i8"); // I×K
    auto *tB = zerogod::ggml_cast_tensor<int8_t>(ctx->tmp_ctx, src1, true, ".i8", true); // K×J

    // 2. bias
    ggml_tensor *tD = nullptr;
    if (bias)
        tD = zerogod::ggml_cast_tensor<int32_t>(ctx->tmp_ctx, bias, true, ".i32", false, J_pad);

    // 3. 출력 버퍼 패딩
    const size_t row_pad_bytes = zerogod::align_up(J_pad * sizeof(int32_t), zerogod::GEMMINI_ROW_ALIGN);
    const size_t stride_e_C = row_pad_bytes / sizeof(int32_t);
    auto *tC = ggml_new_tensor_2d(ctx->tmp_ctx, GGML_TYPE_I32, stride_e_C, I);  // J×I

    memset(tC->data, 0, row_pad_bytes * I);

    // stride
    const size_t sA = tA->nb[1] / sizeof(elem_t);       // == align_up(K,16)
    const size_t sB = tB->nb[1] / sizeof(elem_t);       // == align_up(K,16)
    const size_t sC = stride_e_C;                       // 16 배수
    const size_t sC_CPU = sC * sizeof(int32_t) / sizeof(elem_t);

    GGML_ASSERT(sA % 16 == 0);
    GGML_ASSERT(sB % 16 == 0);
    GGML_ASSERT(sC % 16 == 0);

    // 4. bias 파라미터 준비
    std::vector<int32_t> zero_bias(J_pad, 0);
    const int32_t *D = bias ? (int32_t *)tD->data : zero_bias.data();
    const size_t   sD = bias ? tD->nb[1] / sizeof(int32_t) : 0;
    const bool repeating = bias ? (bias->ne[1] == 1) : true;

    DBG0("    calling tiled_matmul_auto: ptrA=%p ptrB=%p ptrD=%p ptrC=%p\n",
           (void*)tA->data, (void*)tB->data, (void*)D, (void*)tC->data);
    DBG0("    strides: sA=%zu, sB=%zu, sC=%zu, sD=%zu, rep=%d\n",
           sA, sB, sC, sD, repeating);


    // 5. Gemmini 호출 HW
    tiled_matmul_auto(I, J_pad, K,
                      (elem_t*)tA->data,
                      (elem_t*)tB->data,
                      (void*)D,
                      (elem_t*)tC->data,
                      sA, sB, sD, sC_CPU,
                      1.f, 1.f, 1.f,
                      NO_ACTIVATION,
                      1, 1,
                      repeating,
                      false,    // transpose_A
                      false,    // transpose_B
                      false, false,
                      0, CPU);

    // 6. int32 -> float 결과 복사 (stride 사용)
    float   *out_f = (float*)dst->data;
    int32_t *acc32 = (int32_t*)tC->data;
    const size_t row_stride  = stride_e_C;

    for (size_t r = 0; r < I; ++r)
        memcpy(out_f + r * J,
               acc32 + r * row_stride,
               J * sizeof(float));
}

static void ggml_backend_gemmini_out_prod(ggml_backend_gemmini_context *ctx, struct ggml_tensor *dst)
{
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

// backend interface

static const char *ggml_backend_gemmini_get_name(ggml_backend_t backend)
{
    return "GEMMINI";

    GGML_UNUSED(backend);
}

static void ggml_backend_gemmini_free(ggml_backend_t backend)
{
    ggml_backend_gemmini_context *ctx = (ggml_backend_gemmini_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_gemmini_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph)
{
    ggml_backend_gemmini_context *ctx = (ggml_backend_gemmini_context *)backend->context;

    // (1) bias_map 채우기
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto *node = cgraph->nodes[i];
        if (node->op == GGML_OP_ADD && node->src[0]->op == GGML_OP_MUL_MAT) 
            ctx->bias_map[node->src[0]] = node->src[1];
    }

    // (2) tmp_ctx 크기 계산
    if (!ctx->tmp_ctx_initialized)
    {
        size_t total_bytes = 0, total_meta = 0;
        std::set<ggml_tensor *> qset; // 텐서 중복 확인

        // 메모리 계산
        zerogod::ggml_calc_tmp_ctx_size(cgraph, ctx, sizeof(int8_t), qset, total_bytes, total_meta);

        size_t ctx_bytes = total_bytes + total_meta + GGML_MEM_ALIGN;
        posix_memalign(&ctx->arena, 16, ctx_bytes);

        struct ggml_init_params ip = {
            /* .mem_size   = */ ctx_bytes,
            /* .mem_buffer = */ ctx->arena,
            /* .no_alloc   = */ true,
        };

        ctx->tmp_ctx = ggml_init(ip);
        GGML_ASSERT(ctx->tmp_ctx);
        ctx->tmp_ctx_initialized = true;
    }

    for (int i = 0; i < cgraph->n_nodes; i++)
    {
        struct ggml_tensor *node = cgraph->nodes[i];

        switch (node->op)
        {
        case GGML_OP_MUL_MAT: {
            ggml_tensor *bias = nullptr;
            auto it = ctx->bias_map.find(node);
            if (it != ctx->bias_map.end())
                bias = it->second;

            ggml_backend_gemmini_mul_mat(ctx, node, bias);
            break;
        }
        case GGML_OP_OUT_PROD:
            // ggml_backend_gemmini_out_prod(ctx, node);
            break;

        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;

        default:
            GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
    ctx->bias_map.clear();
    ggml_free(ctx->tmp_ctx);
    ctx->tmp_ctx = nullptr;

    free(ctx->arena);
    ctx->arena = nullptr;

    ctx->tmp_ctx_initialized = false;

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i gemmini_backend_i = {
    /* .get_name                = */ ggml_backend_gemmini_get_name,
    /* .free                    = */ ggml_backend_gemmini_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_gemmini_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_gemmini_guid(void)
{
    static ggml_guid guid = {0x10, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d};
    return &guid;
}

ggml_backend_t ggml_backend_gemmini_init(void)
{
    ggml_backend_gemmini_context *ctx = new ggml_backend_gemmini_context;

    ggml_backend_t backend = new ggml_backend{
        /* .guid      = */ ggml_backend_gemmini_guid(),
        /* .interface = */ gemmini_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_gemmini_reg(), 0),
        /* .context   = */ ctx,
    };

    return backend;
}

// bool ggml_backend_is_gemmini(ggml_backend_t backend) {
//     return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_gemmini_guid());
// }

// device interface

static const char *ggml_backend_gemmini_device_get_name(ggml_backend_dev_t dev)
{
    return "GEMMINI";

    GGML_UNUSED(dev);
}

static const char *ggml_backend_gemmini_device_get_description(ggml_backend_dev_t dev)
{
    return "GEMMINI";

    GGML_UNUSED(dev);
}

static void ggml_backend_gemmini_device_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total)
{
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_gemmini_device_get_type(ggml_backend_dev_t dev)
{
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_gemmini_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props)
{
    props->name = ggml_backend_gemmini_device_get_name(dev);
    props->description = ggml_backend_gemmini_device_get_description(dev);
    props->type = ggml_backend_gemmini_device_get_type(dev);
    ggml_backend_gemmini_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_gemmini_device_init_backend(ggml_backend_dev_t dev, const char *params)
{
    return ggml_backend_gemmini_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_gemmini_device_get_buffer_type(ggml_backend_dev_t dev)
{
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_gemmini_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void *ptr, size_t size, size_t max_tensor_size)
{
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_gemmini_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor *op)
{
    const struct ggml_tensor *src0 = op->src[0];
    const struct ggml_tensor *src1 = op->src[1];

    switch (op->op)
    {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;

    case GGML_OP_MUL_MAT:
    {
        // BLAS usually is only faster for large matrices
        const struct ggml_tensor *src0 = op->src[0];
        const struct ggml_tensor *src1 = op->src[1];

        const int64_t ne10 = src1->ne[0];

        const int64_t ne0 = op->ne[0];
        const int64_t ne1 = op->ne[1];

        // TODO: find the optimal value
        const int64_t min_batch = 32;

        return ggml_is_contiguous(src0) &&
               ggml_is_contiguous(src1) &&
               // src1->type == GGML_TYPE_F32 &&
               // (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
               // (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
               true;
    }

    case GGML_OP_OUT_PROD:
        // return op->src[0]->type == GGML_TYPE_F32 &&
        //        op->src[1]->type == GGML_TYPE_F32 &&
        //        ggml_is_matrix(src0) &&
        //        ggml_is_matrix(src1) &&
        //        ggml_is_contiguous(src0) &&
        //        (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
        //        (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);

    default:
        return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_gemmini_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft)
{
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_gemmini_device_i = {
    /* .get_name             = */ ggml_backend_gemmini_device_get_name,
    /* .get_description      = */ ggml_backend_gemmini_device_get_description,
    /* .get_memory           = */ ggml_backend_gemmini_device_get_memory,
    /* .get_type             = */ ggml_backend_gemmini_device_get_type,
    /* .get_props            = */ ggml_backend_gemmini_device_get_props,
    /* .init_backend         = */ ggml_backend_gemmini_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_gemmini_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_gemmini_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_gemmini_device_supports_op,
    /* .supports_buft        = */ ggml_backend_gemmini_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char *ggml_backend_gemmini_reg_get_name(ggml_backend_reg_t reg)
{
    return "GEMMINI";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_gemmini_reg_get_device_count(ggml_backend_reg_t reg)
{
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_gemmini_reg_get_device(ggml_backend_reg_t reg, size_t index)
{
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_gemmini_device = {
        /* .iface   = */ ggml_backend_gemmini_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_gemmini_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_gemmini_reg_i = {
    /* .get_name         = */ ggml_backend_gemmini_reg_get_name,
    /* .get_device_count = */ ggml_backend_gemmini_reg_get_device_count,
    /* .get_device       = */ ggml_backend_gemmini_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_gemmini_reg(void)
{
    static struct ggml_backend_reg ggml_backend_gemmini_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_gemmini_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_gemmini_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_gemmini_reg)
