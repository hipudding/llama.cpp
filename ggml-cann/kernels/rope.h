#ifndef ROPE_H
#define ROPE_H

#pragma pack(push, 8)
typedef struct {
    int64_t input_ne[4];
    int64_t position_ne[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int n_dims;
    int n_orig_ctx;
    float theta_scale;
    float corr_dims[2];

} rope_param;
#pragma pack(pop)

#endif //ROPE_H