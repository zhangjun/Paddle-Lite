function(lite_cc_test_with_model_and_data TARGET)
    if(NOT WITH_TESTING)
        return()
    endif()

    set(options "")
    set(oneValueArgs MODEL DATA CONFIG ARGS)
    set(multiValueArgs "")
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(DEFINED args_MODEL)
        set(ARGS ${ARGS} --model_dir=${LITE_MODEL_DIR}/${args_MODEL})
    endif()
    if(DEFINED args_DATA)
        set(ARGS ${ARGS} --data_dir=${LITE_MODEL_DIR}/${args_DATA})
    endif()
    if(DEFINED args_CONFIG)
        set(ARGS ${ARGS} --config_dir=${LITE_MODEL_DIR}/${args_CONFIG})
    endif()
    if(DEFINED args_ARGS)
        set(ARGS ${ARGS} ${args_ARGS})
    endif()
    lite_cc_test(${TARGET} SRCS ${TARGET}.cc
        DEPS ${lite_model_test_DEPS} paddle_api_full
        ARM_DEPS ${arm_kernels}
        X86_DEPS ${x86_kernels}
        NPU_DEPS ${npu_kernels} ${npu_bridges}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels} ${huawei_ascend_npu_bridges}
        XPU_DEPS ${xpu_kernels} ${xpu_bridges}
        APU_DEPS ${apu_kernels} ${apu_bridges}
        RKNPU_DEPS ${rknpu_kernels} ${rknpu_bridges}
        BM_DEPS ${bm_kernels} ${bm_bridges}
        MLU_DEPS ${mlu_kernels} ${mlu_bridges}
        METAL_DEPS ${metal_kernels}
        ARGS ${ARGS} SERIAL)
    if(DEFINED args_MODEL)
        add_dependencies(${TARGET} extern_lite_download_${args_MODEL}_tar_gz)
    endif()
    if(DEFINED args_DATA)
        add_dependencies(${TARGET} extern_lite_download_${args_DATA}_tar_gz)
    endif()
    if(DEFINED args_CONFIG)
        add_dependencies(${TARGET} extern_lite_download_${args_CONFIG}_tar_gz)
    endif()
endfunction()

if(LITE_WITH_ARM)
    lite_cc_test(test_transformer_with_mask_fp32_arm SRCS test_transformer_with_mask_fp32_arm.cc
      DEPS ${lite_model_test_DEPS} paddle_api_full
      ARM_DEPS ${arm_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/transformer_with_mask_fp32 SERIAL)
  if(WITH_TESTING)
      add_dependencies(test_transformer_with_mask_fp32_arm extern_lite_download_transformer_with_mask_fp32_tar_gz)
  endif()
endif()

if(LITE_WITH_XPU AND NOT LITE_WITH_XTCL)
    lite_cc_test(test_resnet50_fp32_xpu SRCS test_resnet50_fp32_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
    lite_cc_test(test_ernie_fp32_xpu SRCS test_ernie_fp32_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/ernie)
    lite_cc_test(test_bert_fp32_xpu SRCS test_bert_fp32_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/bert)
    if(WITH_TESTING)
        add_dependencies(test_resnet50_fp32_xpu extern_lite_download_resnet50_tar_gz)
        add_dependencies(test_ernie_fp32_xpu extern_lite_download_ernie_tar_gz)
        add_dependencies(test_bert_fp32_xpu extern_lite_download_bert_tar_gz)
    endif()
    # TODO(miaotianxiang): enable later
    #lite_cc_test(test_fpr_fp32_xpu SRCS test_fpr_fp32_xpu.cc
      #DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      #${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      #ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
    #lite_cc_test(test_mmdnn_fp32_xpu SRCS test_mmdnn_fp32_xpu.cc
      #DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      #${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      #ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
endif()

if(LITE_WITH_RKNPU)
    lite_cc_test(test_mobilenetv1_int8_rknpu SRCS test_mobilenetv1_int8_rknpu.cc
      DEPS ${lite_model_test_DEPS} paddle_api_full
      RKNPU_DEPS ${rknpu_kernels} ${rknpu_bridges}
      ARGS --model_dir=${LITE_MODEL_DIR}/MobilenetV1_full_quant SERIAL)
endif()

if(LITE_WITH_APU)
    lite_cc_test(test_mobilenetv1_int8_apu SRCS test_mobilenetv1_int8_apu.cc
      DEPS ${lite_model_test_DEPS} paddle_api_full
      APU_DEPS ${apu_kernels} ${apu_bridges}
      ARGS --model_dir=${LITE_MODEL_DIR}/MobilenetV1_full_quant SERIAL)
endif()

if(LITE_WITH_METAL)
    lite_cc_test_with_model_and_data(test_mobilenet_v1_fp32_metal MODEL mobilenet_v1 DATA ILSVRC2012_500)
endif()
