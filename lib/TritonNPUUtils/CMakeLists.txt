add_triton_library(MLIRTritonNPUUtils
  Utils.cpp
  InterleaveOptimization.cpp
  DebugUtils.cpp

  LINK_LIBS PUBLIC
  BiShengIRAnnotationDialect
  BiShengIRHIVMDialect
  FlirTritonMemoryAccessAnalysis
  FlirTritonMemoryAccessCore
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  TritonAscendIR
  TritonIR
)
