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
