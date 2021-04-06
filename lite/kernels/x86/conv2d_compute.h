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

#pragma once

#include "lite/core/kernel.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include <string>
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class Conv2d : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  Conv2d() = default;
  ~Conv2d() {}
  virtual void PrepareForRun();
  virtual void Run();
  virtual void ReinitWhenNeeded() {
    if (impl_) {
      impl_->ReInitWhenNeeded();
    }
  }
#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }

  std::string kernel_func_name_{"NotImplForConv2d"};
#endif
  ~Conv2d() {
    if (impl_ != nullptr) {
      delete impl_;
      impl == nullptr;
    }
  }

 private:
  using param_t = operators::ConvParam;
  Tensor input_pack_;
  Tensor input_padding_;
  Tensor filter_pack_;
  Tensor output_pack_;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
