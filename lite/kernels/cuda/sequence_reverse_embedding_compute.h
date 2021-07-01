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
#include "lite/core/kernel.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType Ptype>
class SequenceReverseEmbeddingCompute
    : public KernelLite<TARGET(kCUDA), Ptype> {
 public:
  using param_t = operators::LookupTableParam;

  void Run() override;
  virtual ~SequenceReverseEmbeddingCompute() = default;

 private:
  lite_metal::Tensor lod_info_;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
