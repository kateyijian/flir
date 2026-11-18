add_triton_library(FlirTritonMemoryAccessCore
  OpFoldResultUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
)

add_triton_library(FlirTritonMemoryAccessAnalysis
  MaskAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
  FlirTritonMemoryAccessCore
)
