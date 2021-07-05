#pragma once
#include "paddle_lite_factory_helper.h"

USE_LITE_KERNEL(io_copy_once, kMetal, kFloat, kMetalTexture2DArray, host_to_device_image);
USE_LITE_KERNEL(fetch, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(pad2d, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(feed, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(elementwise_add, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(conv2d, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(reshape2, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(cast, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(matmul, kMetal, kFloat, kMetalTexture2DArray, def);