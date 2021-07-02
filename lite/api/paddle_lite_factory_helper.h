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

/*
 * This file defines some MACROS that explicitly determine the op, kernel, mir
 * passes used in the inference lib.
 */
#pragma once

// some platform-independent defintion

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

#define USE_LITE_OP(op_type__)       \
  extern int touch_op_metal_##op_type__(); \
  int LITE_OP_REGISTER_FAKE(op_type__) UNUSED = touch_op_metal_##op_type__();

#define USE_LITE_KERNEL(op_type__, target__, precision__, layout__, alias__) \
  extern int touch_metal_##op_type__##target__##precision__##layout__##alias__();  \
  int op_type_metal__##target__##precision__##layout__##alias__##__use_lite_kernel \
      UNUSED = touch_metal_##op_type__##target__##precision__##layout__##alias__();

#define USE_MIR_PASS(name__)                      \
  extern bool mir_pass_registry##name__##_fake(); \
  static bool mir_pass_usage##name__ UNUSED =     \
      mir_pass_registry##name__##_fake();

#define LITE_OP_REGISTER_FAKE(op_type__) op_type_metal__##__registry__
