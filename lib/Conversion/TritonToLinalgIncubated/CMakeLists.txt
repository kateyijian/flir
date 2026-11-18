if(FLAGTREE_TLE_STRUCT AND TARGET TleDSAToLinalg)
  set(TleToLinalgLib TleDSAToLinalg)
endif()

set(TritonToLinalgIncubatedSources
        TritonToLinalgPass.cpp
        LoadStoreConverter.cpp
        FunctionConverter.cpp
        ArgMinMaxConverter.cpp
        TritonOpConverter.cpp
        HoistBroadcast.cpp
        BlockPtrAnalysis.cpp
        UseAnalysis.cpp
        ImplicitPermute.cpp
        DescriptorConverter.cpp
        MarkTensorKindPass.cpp
        CanonicalizeDebugLocationsPass.cpp
        DeduplicateDebugNopsPass.cpp
        DevicePrintOffsetRewrite.cpp)

set(TritonToLinalgIncubatedLinkLibs
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonAnalysis
  FlirTritonMemoryAccessAnalysis
  MLIRSCFTransforms
  MLIRLinalgTransforms
  ${TleToLinalgLib}
)

# Phase A: the Ascend-specific helpers still live in third_party/ascend.
# They are linked from third_party/ascend/CMakeLists.txt after the Ascend
# libraries have been defined, so this target can be created early enough for
# DiscreteMaskAccessConversion to link it.

add_triton_library(TritonToLinalgIncubated
  ${TritonToLinalgIncubatedSources}

  DEPENDS
  TritonToLinalgIncubatedConversionPassIncGen

  LINK_LIBS PUBLIC
  ${TritonToLinalgIncubatedLinkLibs}
)

if(FLAGTREE_BACKEND STREQUAL "ascend")
  target_include_directories(TritonToLinalgIncubated PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/ascend/include
    ${PROJECT_BINARY_DIR}/third_party/ascend/include
    ${ASCENDNPU_IR_SRC_DIR}/bishengir/include
    ${ASCENDNPU_IR_BINARY_DIR}/bishengir/include)
endif()
