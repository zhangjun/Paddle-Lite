# prepare
modify fbs
```
third-party/flatbuffers/pre-build
lite/model_parser/flatbuffers/framework.fbs
lite/model_parser/flatbuffers/param.fbs

```
#  strip
./lite/tools/build.sh build_optimize_tool
./lite/tools/build_ios_metal_by_models.sh ${model_dirs}

