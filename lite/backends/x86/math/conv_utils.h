// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <immintrin.h>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void transpose8_ps(__m256& row0,   // NOLINT
                   __m256& row1,   // NOLINT
                   __m256& row2,   // NOLINT
                   __m256& row3,   // NOLINT
                   __m256& row4,   // NOLINT
                   __m256& row5,   // NOLINT
                   __m256& row6,   // NOLINT
                   __m256& row7);  // NOLINT
// for input and filter pack
void pack8_m256(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter);
void pack4_m128(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter);

void transform_filter(lite::Tensor* input, lite::Tensor* output);

// for output unpack
void unpack8_m256(lite::Tensor* input, lite::Tensor* output);
void unpack4_m128(lite::Tensor* input, lite::Tensor* output);

// for input padding
void padding8_m256(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding4_m128(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding1_float(lite::Tensor* input,
                    lite::Tensor* output,
                    const std::vector<int>& paddings);

void pack_padding8_m256(lite::Tensor* input,
                        lite::Tensor* output,
                        const int channel_num,
                        const std::vector<int>& paddings);

// for activation - only support relu, relu6
__m256 activation8_m256(__m256 input, const lite_api::ActivationType act_type);
__m128 activation4_m128(__m128 input, const lite_api::ActivationType act_type);
float activation1_float(float input, const lite_api::ActivationType act_type);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
