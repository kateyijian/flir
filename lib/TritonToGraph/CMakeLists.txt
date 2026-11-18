add_triton_library(FlirTritonToGraphLegacy
  LayoutMemoryCompatibilityPass.cpp
  LegacyMemoryAccess/ChunkCoalescing.cpp
  LegacyMemoryAccess/RowCoalescing.cpp
  LegacyMemoryAccess/StridedAxisCoalescing.cpp
  LegacyMemoryAccess/StridedLoadStoreRewrite.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TritonAscendIR
  TritonIR
  TritonToStructured
  MLIRTritonNPUUtils
  FlirTritonMemoryAccessAnalysis
)
