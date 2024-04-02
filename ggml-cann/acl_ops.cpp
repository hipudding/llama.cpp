#include "acl_ops.h"

OpCaller::OpCaller() { attrs = aclopCreateAttr(); }

OpCaller::~OpCaller() {
    for (aclTensorDesc* desc : input_descs) {
        aclDestroyTensorDesc(desc);
    }
    for (aclDataBuffer* buffer : input_buffers) {
        aclDestroyDataBuffer(buffer);
    }
    for (aclTensorDesc* desc : output_descs) {
        aclDestroyTensorDesc(desc);
    }
    for (aclDataBuffer* buffer : output_buffers) {
        aclDestroyDataBuffer(buffer);
    }
    // TODO: may free before use.
    for (void* ptr : ptrs) {
        aclrtFree(ptr);
    }
    aclopDestroyAttr(attrs);
}

OpCaller& OpCaller::name(std::string _op_name) {
    op_name = _op_name;
    return *this;
}

OpCaller& OpCaller::input_no_contiguous(ggml_tensor* tensor, const char* name) {
    aclDataType dtype = type_mapping(tensor->type);
    // TODO
    int64_t ne[] = {tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]};
    aclTensorDesc* tensor_desc =
        aclCreateTensorDesc(dtype, GGML_MAX_DIMS, ne, ACL_FORMAT_ND);
    aclSetTensorDescName(tensor_desc, name);
    input_descs.push_back(tensor_desc);
    aclDataBuffer* data_buffer =
        aclCreateDataBuffer(tensor->data, ggml_nbytes(tensor));
    input_buffers.push_back(data_buffer);
    return *this;
}

OpCaller& OpCaller::input(ggml_tensor* tensor, const char* name) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    return input_no_contiguous(tensor, name);
}

OpCaller& OpCaller::output(ggml_tensor* tensor, const char* name) {
    aclDataType dtype = type_mapping(tensor->type);
    aclTensorDesc* tensor_desc =
        aclCreateTensorDesc(dtype, GGML_MAX_DIMS, tensor->ne, ACL_FORMAT_ND);
    aclSetTensorDescName(tensor_desc, name);
    output_descs.push_back(tensor_desc);
    aclDataBuffer* data_buffer =
        aclCreateDataBuffer(tensor->data, ggml_nbytes(tensor));
    output_buffers.push_back(data_buffer);
    return *this;
}

OpCaller& OpCaller::attr(int64_t value, const char* name) {
    ACL_CHECK(aclopSetAttrInt(attrs, name, value));
    return *this;
}

OpCaller& OpCaller::attr(bool value, const char* name) {
    ACL_CHECK(aclopSetAttrBool(attrs, name, value));
    return *this;
}

OpCaller& OpCaller::attr(float value, const char* name) {
    ACL_CHECK(aclopSetAttrFloat(attrs, name, value));
    return *this;
}

OpCaller& OpCaller::run(aclrtStream stream) {
    ACL_CHECK(aclSetCompileopt(ACL_OP_JIT_COMPILE, "disable"));
    ACL_CHECK(aclopCompileAndExecute(
        op_name.c_str(), input_descs.size(), input_descs.data(),
        input_buffers.data(), output_buffers.size(), output_descs.data(),
        output_buffers.data(), attrs, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr,
        stream));
    return *this;
}

void ggml_cann_cont(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    int64_t src_stride[GGML_MAX_DIMS];
    int64_t dst_stride[GGML_MAX_DIMS];

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        src_stride[i] = src->nb[i] / ggml_type_size(src->type);
        dst_stride[i] = dst->nb[i] / ggml_type_size(src->type);
    }

    int64_t storage_offset[] = {0};
    int64_t storage_offset_dim[] = {1};
    int64_t size_stride_dim[] = {GGML_MAX_DIMS};

    OpCaller op;
    op.name("ViewCopy")
        .input_no_contiguous(dst, "dst")
        .input(dst->ne, ACL_INT64, 1, size_stride_dim, "dst_size", ctx.stream())
        .input(dst_stride, ACL_INT64, 1, size_stride_dim, "dst_stride",
               ctx.stream())
        .input(storage_offset, ACL_INT64, 1, storage_offset_dim,
               "dst_storage_offset", ctx.stream())
        .input_no_contiguous(src, "src")
        .input(src->ne, ACL_INT64, 1, size_stride_dim, "src_size", ctx.stream())
        .input(src_stride, ACL_INT64, 1, size_stride_dim, "src_stride",
               ctx.stream())
        .input(storage_offset, ACL_INT64, 1, storage_offset_dim,
               "src_storage_offset", ctx.stream())
        .output(dst, "dst")
        .run(ctx.stream());
    //aclrtSynchronizeStream(ctx.stream());
}

void ggml_cann_pad(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    int64_t paddings[] = {
        0, dst->ne[3] - src->ne[3], 0, dst->ne[2] - src->ne[2],
        0, dst->ne[1] - src->ne[1], 0, dst->ne[0] - src->ne[0]};
    int64_t dim[] = {GGML_MAX_DIMS, 2};
    OpCaller op;
    op.name("Pad")
        .input(src, "x")
        .input(paddings, ACL_INT64, 2, dim, "paddings", ctx.stream())
        .output(dst, "y")
        .run(ctx.stream());
    //aclrtSynchronizeStream(ctx.stream());
}
