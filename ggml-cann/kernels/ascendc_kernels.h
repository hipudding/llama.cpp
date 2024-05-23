#ifndef ASCENDC_KERNELS_H
#define ASCENDC_KERNELS_H

#include "aclrtlaunch_ascendc_get_row_f32.h"
#include "aclrtlaunch_ascendc_get_row_f16.h"
#include "aclrtlaunch_ascendc_get_row_q8_0.h"
#include "aclrtlaunch_ascendc_get_row_q4_0.h"

#include "aclrtlaunch_ascendc_quantize_f32_q8_0.h"
#include "aclrtlaunch_ascendc_quantize_f16_q8_0.h"

#include "aclrtlaunch_ascendc_rope_init_cache.h"
#include "aclrtlaunch_ascendc_dup_by_rows.h"
#include "rope.h"

#endif  // ASCENDC_KERNELS_H