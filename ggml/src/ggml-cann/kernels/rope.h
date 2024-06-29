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
    float corr_dims0;
    float corr_dims1;
    float corr_dims0_neg;
    float rope_yarn_ramp_max_inv;
    bool is_neox;
    bool use_freq_factors;

} rope_param;
#pragma pack(pop)

#endif //ROPE_H