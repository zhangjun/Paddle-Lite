// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/flatten_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace xpu {

void FlattenContiguousRangeCompute::Run() {
  auto& param = Param<operators::FlattenContiguousRangeParam>();
  auto x = param.x;
  auto out = param.out;
  auto out_dims = out->dims();
  out->ShareDataWith(*x);
  out->Resize(out_dims);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(flatten_contiguous_range,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite_metal::kernels::xpu::FlattenContiguousRangeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
