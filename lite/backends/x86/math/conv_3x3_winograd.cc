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

// #if defined(__AVX__)
#include <immintrin.h>
// #endif
#include <vector>
#include "lite/backends/x86/math/conv_3x3_winograd.h"
#include "lite/backends/x86/math/conv_utils.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

/**
 * \brief transpose with arm neon optimization
 * @param data_out
 * @param data_in
 * @param w_in
 * @param h_in
 */
static void transpose(float* data_out,
                      const float* data_in,
                      int w_in,
                      int h_in) {
  for (int j = 0; j < h_in; ++j) {
    for (int i = 0; i < w_in; ++i) {
      data_out[i * h_in + j] = data_in[j * w_in + i];
    }
  }
}

/**
* \brief winograd transform conv3x3 weights, f63
* this is done in op initialization or creation, only do once
* dout = G * g * GT, where G is the transform coeff, g is the input weights
* @param dout
* @param din
* @param ch_out
* @param ch_in
* @param work_space
*/
static void winograd_transform_weights(
    float* dout, const float* din, int ch_out, int ch_in, float* work_space) {
  const float coeff[8][3] = {{1.0f, 0.0f, 0.0f},
                             {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                             {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                             {1.0f / 90, 1.0f / 45, 2.0f / 45},
                             {1.0f / 90, -1.0f / 45, 2.0f / 45},
                             {32.0f / 45, 16.0f / 45, 8.0f / 45},
                             {32.0f / 45, -16.0f / 45, 8.0f / 45},
                             {0.0f, 0.0f, 1.0f}};

  float* ptr_out = static_cast<float*>(work_space);

  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const float* kernel0 =
          static_cast<const float*>(din) + (i * ch_in + j) * 9;
      float* ptr_channel = ptr_out + (i * ch_in + j) * 64;

      //! transform kernel, transposed
      const float* k0 = kernel0;
      const float* k1 = kernel0 + 3;
      const float* k2 = kernel0 + 6;

      //! h
      float tmp[8][3];

      for (int i = 0; i < 8; i++) {
        tmp[i][0] =
            k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
        tmp[i][1] =
            k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
        tmp[i][2] =
            k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
      }

      //! v
      for (int j = 0; j < 8; j++) {
        float* tmpp = &tmp[j][0];

        for (int i = 0; i < 8; i++) {
          ptr_channel[j * 8 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
        }
      }
    }
  }

  transpose(static_cast<float*>(dout), ptr_out, 64, ch_out * ch_in);
}

static inline void winograd_f6k3_input_inplace_avx2(__m256& m0,  // NOLINT
                                                    __m256& m1,  // NOLINT
                                                    __m256& m2,  // NOLINT
                                                    __m256& m3,  // NOLINT
                                                    __m256& m4,  // NOLINT
                                                    __m256& m5,  // NOLINT
                                                    __m256& m6,  // NOLINT
                                                    __m256& m7   // NOLINT
                                                    ) {
  const __m256 m_5p25 = _mm256_set1_ps(5.25f);
  const __m256 m_4p25 = _mm256_set1_ps(4.25f);
  const __m256 m_4p0 = _mm256_set1_ps(4.f);
  const __m256 m_2p5 = _mm256_set1_ps(2.5f);
  const __m256 m_2p0 = _mm256_set1_ps(2.f);
  const __m256 m_1p25 = _mm256_set1_ps(1.25f);
  const __m256 m_0p5 = _mm256_set1_ps(0.5f);
  const __m256 m_0p25 = _mm256_set1_ps(0.25f);
  m0 = m0 - m6 + (m4 - m2) * m_5p25;
  m7 = m7 - m1 + (m3 - m5) * m_5p25;

  __m256 t1 = m2 + m6 - m4 * m_4p25;
  __m256 t2 = m1 + m5 - m3 * m_4p25;

  __m256 s1 = m4 * m_1p25;
  __m256 s2 = m3 * m_2p5;

  __m256 p1 = m6 + (m2 * m_0p25 - s1);
  __m256 p2 = m1 * m_0p5 - s2 + m5 * m_2p0;

  m3 = p1 + p2;
  m4 = p1 - p2;

  p1 = m6 + (m2 - s1) * m_4p0;
  p2 = m1 * m_2p0 - s2 + m5 * m_0p5;

  m5 = p1 + p2;
  m6 = p1 - p2;

  m1 = _mm256_add_ps(t1, t2);
  m2 = _mm256_sub_ps(t1, t2);

  transpose8_ps(m0, m1, m2, m3, m4, m5, m6, m7);

  m0 = m0 - m6 + (m4 - m2) * m_5p25;
  m7 = m7 - m1 + (m3 - m5) * m_5p25;

  t1 = m2 + m6 - m4 * m_4p25;
  t2 = m1 + m5 - m3 * m_4p25;

  s1 = m4 * m_1p25;
  s2 = m3 * m_2p5;

  p1 = m6 + (m2 * m_0p25 - s1);
  p2 = m1 * m_0p5 - s2 + m5 * m_2p0;

  m3 = p1 + p2;
  m4 = p1 - p2;

  p1 = m6 + (m2 - s1) * m_4p0;
  p2 = m1 * m_2p0 - s2 + m5 * m_0p5;

  m5 = p1 + p2;
  m6 = p1 - p2;

  m1 = _mm256_add_ps(t1, t2);
  m2 = _mm256_sub_ps(t1, t2);
}

static inline void winograd_f6k3_output_inplace_avx2_float_in(
    __m256& m0,  // NOLINT
    __m256& m1,  // NOLINT
    __m256& m2,  // NOLINT
    __m256& m3,  // NOLINT
    __m256& m4,  // NOLINT
    __m256& m5,  // NOLINT
    __m256& m6,  // NOLINT
    __m256& m7,  // NOLINT
    float* din,
    const float& bias,
    const bool& with_relu) {
  const __m256 m_32p0 = _mm256_set1_ps(32.f);
  const __m256 m_16p0 = _mm256_set1_ps(16.f);
  const __m256 m_8p0 = _mm256_set1_ps(8.f);
  const __m256 m_4p0 = _mm256_set1_ps(4.f);
  const __m256 m_2p0 = _mm256_set1_ps(2.f);

  const __m256 m_0p5 = _mm256_set1_ps(0.5f);
  const __m256 m_0p25 = _mm256_set1_ps(0.25f);
  const __m256 m_0p125 = _mm256_set1_ps(0.125f);
  const __m256 m_0p0625 = _mm256_set1_ps(0.0625f);
  const __m256 m_0p03125 = _mm256_set1_ps(0.03125f);

  m0 = _mm256_loadu_ps(&din[0 * 8]);
  m1 = _mm256_loadu_ps(&din[1 * 8]);
  m2 = _mm256_loadu_ps(&din[2 * 8]);
  m3 = _mm256_loadu_ps(&din[3 * 8]);
  m4 = _mm256_loadu_ps(&din[4 * 8]);
  m5 = _mm256_loadu_ps(&din[5 * 8]);
  m6 = _mm256_loadu_ps(&din[6 * 8]);
  m7 = _mm256_loadu_ps(&din[7 * 8]);

  __m256 m1_add_m2 = m1 + m2;
  __m256 m1_sub_m2 = m1 - m2;
  __m256 m3_add_m4 = m3 + m4;
  __m256 m3_sub_m4 = m3 - m4;
  __m256 m5_add_m6 = m5 + m6;
  __m256 m5_sub_m6 = m5 - m6;

  // Finised with M[0-6] as **inputs** here.
  m0 = m0 + m1_add_m2 + m3_add_m4 + m5_add_m6;
  m2 = m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6;
  m4 = m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625;
  m1 = m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5;
  m3 = m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125;
  m5 = m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125;
  m6 = _mm256_setzero_ps();
  m7 = _mm256_setzero_ps();

  transpose8_ps(m0, m1, m2, m3, m4, m5, m6, m7);

  m1_add_m2 = m1 + m2;
  m1_sub_m2 = m1 - m2;
  m3_add_m4 = m3 + m4;
  m3_sub_m4 = m3 - m4;
  m5_add_m6 = m5 + m6;
  m5_sub_m6 = m5 - m6;

  const __m256 bias_value = _mm256_set1_ps(bias);
  const __m256 m_0p0 = _mm256_setzero_ps();

  if (with_relu) {
    m0 = _mm256_max_ps(bias_value + m0 + m1_add_m2 + m3_add_m4 + m5_add_m6,
                       m_0p0);
    m2 = _mm256_max_ps(
        bias_value + m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6, m_0p0);
    m4 = _mm256_max_ps(
        bias_value + m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625,
        m_0p0);
    m1 = _mm256_max_ps(
        bias_value + m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5, m_0p0);
    m3 = _mm256_max_ps(
        bias_value + m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125,
        m_0p0);
    m5 = _mm256_max_ps(bias_value + m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 +
                           m5_sub_m6 * m_0p03125,
                       m_0p0);
  } else {
    m0 = bias_value + m0 + m1_add_m2 + m3_add_m4 + m5_add_m6;
    m2 = bias_value + m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6;
    m4 = bias_value + m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625;
    m1 = bias_value + m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5;
    m3 = bias_value + m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125;
    m5 = bias_value + m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 +
         m5_sub_m6 * m_0p03125;
  }
}

static inline void winograd_f6k3_output_inplace_avx2(__m256& m0,  // NOLINT
                                                     __m256& m1,  // NOLINT
                                                     __m256& m2,  // NOLINT
                                                     __m256& m3,  // NOLINT
                                                     __m256& m4,  // NOLINT
                                                     __m256& m5,  // NOLINT
                                                     __m256& m6,  // NOLINT
                                                     __m256& m7,  // NOLINT
                                                     const float& bias) {
  const __m256 m_32p0 = _mm256_set1_ps(32.f);
  const __m256 m_16p0 = _mm256_set1_ps(16.f);
  const __m256 m_8p0 = _mm256_set1_ps(8.f);
  const __m256 m_4p0 = _mm256_set1_ps(4.f);
  const __m256 m_2p0 = _mm256_set1_ps(2.f);

  const __m256 m_0p5 = _mm256_set1_ps(0.5f);
  const __m256 m_0p25 = _mm256_set1_ps(0.25f);
  const __m256 m_0p125 = _mm256_set1_ps(0.125f);
  const __m256 m_0p0625 = _mm256_set1_ps(0.0625f);
  const __m256 m_0p03125 = _mm256_set1_ps(0.03125f);

  __m256 m1_add_m2 = m1 + m2;
  __m256 m1_sub_m2 = m1 - m2;
  __m256 m3_add_m4 = m3 + m4;
  __m256 m3_sub_m4 = m3 - m4;
  __m256 m5_add_m6 = m5 + m6;
  __m256 m5_sub_m6 = m5 - m6;

  // Finised with M[0-6] as **inputs** here.
  m0 = m0 + m1_add_m2 + m3_add_m4 + m5_add_m6;
  m2 = m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6;
  m4 = m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625;
  m1 = m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5;
  m3 = m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125;
  m5 = m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125;
  m6 = _mm256_setzero_ps();
  m7 = _mm256_setzero_ps();

  transpose8_ps(m0, m1, m2, m3, m4, m5, m6, m7);

  m1_add_m2 = m1 + m2;
  m1_sub_m2 = m1 - m2;
  m3_add_m4 = m3 + m4;
  m3_sub_m4 = m3 - m4;
  m5_add_m6 = m5 + m6;
  m5_sub_m6 = m5 - m6;

  const __m256 bias_value = _mm256_set1_ps(bias);

  m0 = bias_value + m0 + m1_add_m2 + m3_add_m4 + m5_add_m6;
  m2 = bias_value + m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6;
  m4 = bias_value + m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625;
  m1 = bias_value + m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5;
  m3 = bias_value + m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125;
  m5 = bias_value + m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125;
}

void conv_3x3s1_winograd_m256(lite::Tensor* input,
                              lite::Tensor* output,
                              lite::Tensor* filter,
                              lite::Tensor* bias,
                              const bool has_act,
                              const lite_api::ActivationType act_type,
                              const std::vector<int>& paddings) {
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int input_channel = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  CHECK_EQ(filter->dims().size(), 6UL);
  const int kernel_h = filter->dims()[2];
  const int kernel_w = filter->dims()[3];
  const float* filter_data = filter->data<float>();
  const int filter_kernel_size = kernel_h * kernel_w;

  const int output_channel = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  float* output_data = output->mutable_data<float>();

  const int tile_w = (output_width + 5) / 6;
  const int tile_h = (output_height + 5) / 6;
  // const int size_tile = tile_w * tile_h;

  const int pad_h = paddings[0];
  const int pad_w = paddings[2];

  const int input_channel_step = input_height * input_width;
  const int output_channel_step = output_height * output_width;
  // const int trans_channel_step = 8 * 8 * size_tile;

  // int max_ch = input_channel > output_channel ? input_channel :
  // output_channel;

  for (int bs = 0; bs < batch_size; ++bs) {
    const float* din_batch =
        input_data + bs * input_channel * input_channel_step;
    float* dout_batch = output_data + bs * output_channel * output_channel_step;

    for (int oc = 0; oc < output_channel; oc++) {
      for (int h = 0; h < tile_h; h++) {
        for (int w = 0; w < tile_w; w++) {
          __m256 result[8] = {_mm256_setzero_ps()};

          for (int ic = 0; ic < input_channel; ++ic) {
            //! prepare data 8x8
            //! row 8
            __m256 data_in_tmp[8] = {_mm256_setzero_ps()};
            const float* din_channel = din_batch + ic * input_channel_step;

            // memset(data_in_tmp[0], 0, sizeof(float) * 64);
            for (int j = 0; j < 8; ++j) {
              int start_row = h * 6 + j - pad_h;

              if (start_row >= 0 && start_row < input_height) {
                for (int k = 0; k < 8; ++k) {
                  int start_col = w * 6 + k - pad_w;

                  if (start_col >= 0 && start_col < input_width) {
                    data_in_tmp[j][k] =
                        din_channel[start_row * input_width + start_col];
                  }
                }
              }
            }

            winograd_f6k3_input_inplace_avx2(data_in_tmp[0],
                                             data_in_tmp[1],
                                             data_in_tmp[2],
                                             data_in_tmp[3],
                                             data_in_tmp[4],
                                             data_in_tmp[5],
                                             data_in_tmp[6],
                                             data_in_tmp[7]);

            // exit(0)
            const float* filter_ptr =
                (const float*)filter_data +
                filter_kernel_size * input_channel * oc * 64;
            for (int i = 0; i < 8; i++) {
              result[i] += data_in_tmp[i] * _mm256_loadu_ps(filter_ptr + i * 8);
            }
          }

          float bias_value =
              bias ? bias->data<float>()[bs * output_channel + oc] : 0.f;
          // output
          winograd_f6k3_output_inplace_avx2(result[0],
                                            result[1],
                                            result[2],
                                            result[3],
                                            result[4],
                                            result[5],
                                            result[6],
                                            result[7],
                                            bias_value);

          // act
          for (int i = 0; i < 8; ++i) {
            if (has_act) {
              result[i] = activation8_m256(result[i], act_type);
            }
          }

          float* dout_channel = dout_batch + oc * output_channel_step;

          for (int j = 0; j < 6; ++j) {
            int end_row = h * 6 + j;

            if (end_row < output_height) {
              for (int k = 0; k < 6; ++k) {
                int end_col = w * 6 + k;

                if (end_col < output_width) {
                  dout_channel[end_row * output_width + end_col] = result[j][k];
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
