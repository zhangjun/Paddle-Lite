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

#include "lite/backends/x86/math/conv_3x3_pack8.h"
#include <vector>
#include "lite/backends/x86/math/conv_utils.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void conv_3x3_m256(lite::Tensor* input,
                   lite::Tensor* output,
                   lite::Tensor* filter,
                   lite::Tensor* bias,
                   const int stride_h,
                   const int stride_w,
                   const int dilation_h,
                   const int dilation_w,
                   const bool has_act,
                   const lite_api::ActivationType act_type) {
  // for (int p = 0; p < num_output / out_elempack; p++){
  // input [bs, ic/8, ih, iw, 8]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int input_channel = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  // filter [1, oc/8, kh, kw, 8]
  CHECK_EQ(filter->dims().size(), 6UL);
  const int kernel_h = filter->dims()[2];
  const int kernel_w = filter->dims()[3];
  const float* filter_data = filter->data<float>();

  // output [bs, oc/8, oh, ow, 8]
  CHECK_EQ(output->dims().size(), 5UL);
  const int output_channel = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  float* output_data = output->mutable_data<float>();

  const int input_group_step = input_width * 8;
  const int input_channel_step = input_height * input_width * 8;
  const int input_batch_step = input_channel * input_height * input_width * 8;

  const int output_channel_step = output_height * output_width * 8;
  const int output_batch_step =
      output_channel * output_height * output_width * 8;

  // const int filter_channel_step = kernel_h * kernel_w * 8;

  const int filter_kernel_size = kernel_w * kernel_h;
  // kernel offsets
  std::vector<int> _space_ofs(filter_kernel_size);
  int* space_ofs = &_space_ofs[0];
  {
    int p1 = 0;
    int p2 = 0;
    int gap = input_width * dilation_h - kernel_w * dilation_w;
    for (int i = 0; i < kernel_h; i++) {
      for (int j = 0; j < kernel_w; j++) {
        space_ofs[p1] = p2;
        p1++;
        p2 += dilation_w;
      }
      p2 += gap;
    }
  }

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int oc = 0; oc < output_channel; ++oc) {
      float* output_ptr =
          output_data + bs * output_batch_step + oc * output_channel_step;
      for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
          __m256 _sum = _mm256_set1_ps(0.f);

          if (bias) {
            _sum = _mm256_loadu_ps(bias->data<float>() + oc * 8);
          }

          const float* kptr = (const float*)filter_data +
                              filter_kernel_size * input_channel * oc * 64;

          // channels
          for (int ic = 0; ic < input_channel; ++ic) {
            // const Mat m = bottom_blob_bordered.channel(q);
            // const float* sptr = m.row(h * stride_h) + w * stride_w * 8;
            const float* input_ptr =
                input_data + bs * input_batch_step + ic * input_channel_step;
            const float* sptr =
                input_ptr + h * stride_h * input_group_step + w * stride_w * 8;

            for (int k = 0; k < filter_kernel_size; k++) {
              __m256 _val0 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8));
              __m256 _val1 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 1);
              __m256 _val2 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 2);
              __m256 _val3 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 3);
              __m256 _val4 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 4);
              __m256 _val5 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 5);
              __m256 _val6 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 6);
              __m256 _val7 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 7);

              __m256 _w0 = _mm256_loadu_ps(kptr);
              __m256 _mul0 = _mm256_mul_ps(_val0, _w0);
              __m256 _w1 = _mm256_loadu_ps(kptr + 8);
              __m256 _mul1 = _mm256_mul_ps(_val1, _w1);
              __m256 _w2 = _mm256_loadu_ps(kptr + 16);
              __m256 _mul2 = _mm256_mul_ps(_val2, _w2);
              __m256 _w3 = _mm256_loadu_ps(kptr + 24);
              __m256 _mul3 = _mm256_mul_ps(_val3, _w3);
              __m256 _w4 = _mm256_loadu_ps(kptr + 32);
              __m256 _mul4 = _mm256_mul_ps(_val4, _w4);
              __m256 _w5 = _mm256_loadu_ps(kptr + 40);
              __m256 _mul5 = _mm256_mul_ps(_val5, _w5);
              __m256 _w6 = _mm256_loadu_ps(kptr + 48);
              __m256 _mul6 = _mm256_mul_ps(_val6, _w6);
              __m256 _w7 = _mm256_loadu_ps(kptr + 56);
              __m256 _mul7 = _mm256_mul_ps(_val7, _w7);
              __m256 _sum01 = _mm256_add_ps(_mul0, _mul1);
              __m256 _sum23 = _mm256_add_ps(_mul2, _mul3);
              __m256 _sum45 = _mm256_add_ps(_mul4, _mul5);
              __m256 _sum67 = _mm256_add_ps(_mul6, _mul7);
              __m256 _sum_lo = _mm256_add_ps(_sum01, _sum23);
              __m256 _sum_hi = _mm256_add_ps(_sum45, _sum67);
              __m256 _sum_all = _mm256_add_ps(_sum_lo, _sum_hi);
              _sum = _mm256_add_ps(_sum_all, _sum);

              kptr += 64;
            }
          }
          if (has_act) {
            _sum = activation8_m256(_sum, act_type);
          }
          _mm256_storeu_ps(output_ptr + w * 8, _sum);
        }
        output_ptr += output_width * 8;
      }
    }
  }
}

void conv_3x3_8to4(lite::Tensor* input,
                   lite::Tensor* output,
                   lite::Tensor* filter,
                   lite::Tensor* bias,
                   const int stride_h,
                   const int stride_w,
                   const int dilation_h,
                   const int dilation_w,
                   const bool has_act,
                   const lite_api::ActivationType act_type) {
  // input [bs, ic/8, ih, iw, 8]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int input_channel = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  // filter [1, oc/8, kh, kw, 8]
  CHECK_EQ(filter->dims().size(), 6UL);
  const int kernel_h = filter->dims()[2];
  const int kernel_w = filter->dims()[3];
  const float* filter_data = filter->data<float>();

  // output [bs, oc/8, oh, ow, 8]
  CHECK_EQ(output->dims().size(), 5UL);
  const int output_channel = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  float* output_data = output->mutable_data<float>();

  const int input_group_step = input_width * 8;
  const int input_channel_step = input_height * input_width * 8;
  const int input_batch_step = input_channel * input_height * input_width * 8;

  const int output_channel_step = output_height * output_width * 4;
  const int output_batch_step =
      output_channel * output_height * output_width * 4;

  // const int filter_channel_step = kernel_h * kernel_w * 8;

  const int filter_kernel_size = kernel_w * kernel_h;
  // kernel offsets
  std::vector<int> _space_ofs(filter_kernel_size);
  int* space_ofs = &_space_ofs[0];
  {
    int p1 = 0;
    int p2 = 0;
    int gap = input_width * dilation_h - kernel_w * dilation_w;
    for (int i = 0; i < kernel_h; i++) {
      for (int j = 0; j < kernel_w; j++) {
        space_ofs[p1] = p2;
        p1++;
        p2 += dilation_w;
      }
      p2 += gap;
    }
  }

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int oc = 0; oc < output_channel; ++oc) {
      float* output_ptr =
          output_data + bs * output_batch_step + oc * output_channel_step;
      for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
          __m128 _sum = _mm_set1_ps(0.f);

          if (bias) {
            _sum = _mm_loadu_ps(bias->data<float>() + oc * 4);
          }

          const float* kptr = (const float*)filter_data +
                              filter_kernel_size * input_channel * oc * 32;
          //  const float* kptr = weight_data_packed.channel(p);

          for (int ic = 0; ic < input_channel; ++ic) {
            const float* input_ptr =
                input_data + bs * input_batch_step + ic * input_channel_step;
            const float* sptr =
                input_ptr + h * stride_h * input_group_step + w * stride_w * 8;

            for (int k = 0; k < filter_kernel_size; k++) {
              __m128 _val0 = _mm_broadcast_ss((sptr + space_ofs[k] * 8));
              __m128 _val1 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 1);
              __m128 _val2 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 2);
              __m128 _val3 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 3);
              __m128 _val4 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 4);
              __m128 _val5 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 5);
              __m128 _val6 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 6);
              __m128 _val7 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 7);

              __m128 _w0 = _mm_loadu_ps(kptr);
              _sum = _mm_fmadd_ps(_val0, _w0, _sum);
              __m128 _w1 = _mm_loadu_ps(kptr + 4);
              _sum = _mm_fmadd_ps(_val1, _w1, _sum);
              __m128 _w2 = _mm_loadu_ps(kptr + 8);
              _sum = _mm_fmadd_ps(_val2, _w2, _sum);
              __m128 _w3 = _mm_loadu_ps(kptr + 12);
              _sum = _mm_fmadd_ps(_val3, _w3, _sum);
              __m128 _w4 = _mm_loadu_ps(kptr + 16);
              _sum = _mm_fmadd_ps(_val4, _w4, _sum);
              __m128 _w5 = _mm_loadu_ps(kptr + 20);
              _sum = _mm_fmadd_ps(_val5, _w5, _sum);
              __m128 _w6 = _mm_loadu_ps(kptr + 24);
              _sum = _mm_fmadd_ps(_val6, _w6, _sum);
              __m128 _w7 = _mm_loadu_ps(kptr + 28);
              _sum = _mm_fmadd_ps(_val7, _w7, _sum);

              kptr += 32;
            }
          }

          if (has_act) {
            _sum = activation4_m128(_sum, act_type);
          }
          _mm_storeu_ps(output_ptr + w * 4, _sum);
        }

        output_ptr += output_width * 4;
      }
    }
  }
}

void conv_3x3_4to8(lite::Tensor* input,
                   lite::Tensor* output,
                   lite::Tensor* filter,
                   lite::Tensor* bias,
                   const int stride_h,
                   const int stride_w,
                   const int dilation_h,
                   const int dilation_w,
                   const bool has_act,
                   const lite_api::ActivationType act_type) {
  // input [bs, ic/8, ih, iw, 8]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int input_channel = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  // filter [1, oc/8, kh, kw, 8]
  CHECK_EQ(filter->dims().size(), 6UL);
  const int kernel_h = filter->dims()[2];
  const int kernel_w = filter->dims()[3];
  const float* filter_data = filter->data<float>();

  // output [bs, oc/8, oh, ow, 8]
  CHECK_EQ(output->dims().size(), 5UL);
  const int output_channel = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  float* output_data = output->mutable_data<float>();

  const int input_group_step = input_width * 4;
  const int input_channel_step = input_height * input_width * 4;
  const int input_batch_step = input_channel * input_height * input_width * 4;

  const int output_channel_step = output_height * output_width * 8;
  const int output_batch_step =
      output_channel * output_height * output_width * 8;

  // const int filter_channel_step = kernel_h * kernel_w * 8;

  const int filter_kernel_size = kernel_w * kernel_h;
  // kernel offsets
  std::vector<int> _space_ofs(filter_kernel_size);
  int* space_ofs = &_space_ofs[0];
  {
    int p1 = 0;
    int p2 = 0;
    int gap = input_width * dilation_h - kernel_w * dilation_w;
    for (int i = 0; i < kernel_h; i++) {
      for (int j = 0; j < kernel_w; j++) {
        space_ofs[p1] = p2;
        p1++;
        p2 += dilation_w;
      }
      p2 += gap;
    }
  }

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int oc = 0; oc < output_channel; ++oc) {
      float* output_ptr =
          output_data + bs * output_batch_step + oc * output_channel_step;
      for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
          __m256 _sum = _mm256_set1_ps(0.f);

          if (bias) {
            _sum = _mm256_loadu_ps(bias->data<float>() + oc * 8);
          }

          const float* kptr = (const float*)filter_data +
                              filter_kernel_size * input_channel * oc * 32;
          for (int ic = 0; ic < input_channel; ++ic) {
            const float* input_ptr =
                input_data + bs * input_batch_step + ic * input_channel_step;
            const float* sptr =
                input_ptr + h * stride_h * input_group_step + w * stride_w * 4;

            for (int k = 0; k < filter_kernel_size; k++) {
              __m256 _val0 = _mm256_broadcast_ss((sptr + space_ofs[k] * 4));
              __m256 _val1 = _mm256_broadcast_ss((sptr + space_ofs[k] * 4) + 1);
              __m256 _val2 = _mm256_broadcast_ss((sptr + space_ofs[k] * 4) + 2);
              __m256 _val3 = _mm256_broadcast_ss((sptr + space_ofs[k] * 4) + 3);

              __m256 _w0 = _mm256_loadu_ps(kptr);
              _sum = _mm256_fmadd_ps(_val0, _w0, _sum);
              __m256 _w1 = _mm256_loadu_ps(kptr + 8);
              _sum = _mm256_fmadd_ps(_val1, _w1, _sum);
              __m256 _w2 = _mm256_loadu_ps(kptr + 16);
              _sum = _mm256_fmadd_ps(_val2, _w2, _sum);
              __m256 _w3 = _mm256_loadu_ps(kptr + 24);
              _sum = _mm256_fmadd_ps(_val3, _w3, _sum);

              kptr += 32;
            }
          }

          if (has_act) {
            _sum = activation8_m256(_sum, act_type);
          }
          _mm256_storeu_ps(output_ptr + w * 8, _sum);
        }

        output_ptr += output_width * 8;
      }
    }
  }
  std::cout << "batch_size: " << batch_size << ", " << output_channel
            << std::endl;
  std::cout << "out_h: " << output_height << ", out_w:" << output_width
            << std::endl;
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
