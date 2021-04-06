// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/x86/conv2d_compute.h"
#include "lite/backends/x86/math/conv_3x3_pack8.h"
#include "lite/backends/x86/math/conv_3x3_winograd.h"
#include "lite/backends/x86/math/conv_utils.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <>
void Conv2d<float>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);

  auto input_dims = param.x->dims();
  CHECK_EQ(input_dims.size(), 4UL);
  int batch_size = param.x->dims()[0];
  int input_channel = param.x->dims()[1];
  int output_channel = param.filter->dims()[0];

  const int pack_in =
      input_channel % 8 == 0 ? 8 : input_channel % 4 == 0 ? 4 : 1;
  const int pack_out =
      output_channel % 8 == 0 ? 8 : output_channel % 4 == 0 ? 4 : 1;
  const int pack_num = input_channel / pack_in;
  const int pack_num_out = output_channel / pack_out;

  if (pack_in == 8) {
    lite::x86::math::pack_padding8_m256(
        param.x, &input_padding_, pack_num, *(param.paddings));
  } else if (pack_in == 4) {
    lite::x86::math::pack4_m128(param.x, &input_pack_, pack_num, false);
    lite::x86::math::padding4_m128(
        &input_pack_, &input_padding_, *(param.paddings));
  } else {
    lite::x86::math::padding1_float(
        param.x, &input_padding_, *(param.paddings));
  }

  // filter [oc, ic/groups=1, kh, kw]
  auto filter_dims = param.filter->dims();
  CHECK_EQ(filter_dims.size(), 4UL);
  int kernel_h = param.filter->dims()[2];
  int kernel_w = param.filter->dims()[3];

  // filter [oc, ic, ih, iw] & pack_in=8, pack_out=8 => [oc/8, ic/8, ih, iw, 8,
  // 8]
  // filter [oc, ic, ih, iw] & pack_in=4, pack_out=4 => [ic/4, ic/4, ih, iw, 4,
  // 4]
  filter_pack_.Resize(
      {pack_num_out, pack_num, kernel_h, kernel_w, pack_in, pack_out});
  lite::x86::math::transform_filter(param.filter, &filter_pack_);

  // attributes
  const int stride_h = param.strides[0];
  const int stride_w = param.strides[1];
  const int dilation_h = (*param.dilations)[0];
  const int dilation_w = (*param.dilations)[1];

  // act type
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  auto act_type = act_param.active_type;

  // output [bs, oc, oh, ow]
  CHECK_EQ(param.output->dims().size(), 4UL);
  const int in_h = input_padding_.dims()[2], in_w = input_padding_.dims()[3];
  const int kernel_extend_h = dilation_h * (kernel_h - 1) + 1;
  const int kernel_extend_w = dilation_w * (kernel_w - 1) + 1;
  int output_height = (in_h - kernel_extend_h) / stride_h + 1;
  int output_width = (in_w - kernel_extend_w) / stride_w + 1;
  // output_trans [bs, oc/8, oh, ow, 8]
  // output_trans [bs, oc/4, oh, ow, 4]
  output_pack_.Resize(
      {batch_size, pack_num_out, output_height, output_width, pack_out});

  if (pack_in == 8 && pack_out == 8) {
    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1) {
      lite::x86::math::conv_3x3s1_winograd_m256(&input_padding_,
                                                &output_pack_,
                                                &filter_pack_,
                                                param.bias,
                                                has_act,
                                                act_type,
                                                *(param.paddings));
    } else {
      lite::x86::math::conv_3x3_m256(&input_padding_,
                                     &output_pack_,
                                     &filter_pack_,
                                     param.bias,
                                     stride_h,
                                     stride_w,
                                     dilation_h,
                                     dilation_w,
                                     has_act,
                                     act_type);
    }
  }
  if (pack_in == 8 && pack_out == 4) {
    lite::x86::math::conv_3x3_8to4(&input_padding_,
                                   &output_pack_,
                                   &filter_pack_,
                                   param.bias,
                                   stride_h,
                                   stride_w,
                                   dilation_h,
                                   dilation_w,
                                   has_act,
                                   act_type);
  }
  if (pack_in == 4 && pack_out == 8) {
    lite::x86::math::conv_3x3_4to8(&input_padding_,
                                   &output_pack_,
                                   &filter_pack_,
                                   param.bias,
                                   stride_h,
                                   stride_w,
                                   dilation_h,
                                   dilation_w,
                                   has_act,
                                   act_type);
  }
}

#ifdef LITE_WITH_PROFILE
template <>
void Conv2d<float>::SetProfileRuntimeKernelInfo(
    paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
