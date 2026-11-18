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
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
  ${TleToLinalgLib}
)

# Phase A: the Ascend-specific helpers still live in third_party/ascend.
# When this target is built as part of the Ascend backend, link them here.
if(FLAGTREE_BACKEND STREQUAL "ascend")
  list(APPEND TritonToLinalgIncubatedLinkLibs
    BiShengIRHIVMDialect
    BiShengIRHFusionDialect
    BiShengIRAnnotationDialect
    BiShengIRScopeDialect
    TritonAscendIR
    TritonStructuredIR
    TritonMemoryAccessAnalysis
    TritonToStructured
    TritonToUnstructure
    TritonToGraph
    TritonControlFlowOpt
  )
endif()

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
