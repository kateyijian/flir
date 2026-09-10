// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

// Mask `(offs - 128) >= 0` has a non-zero lower bound: the valid region is
// the suffix [128, 256). The prefix-only mask-dims model ([0, dims)) cannot
// express it, so MaskAnalysis must fail and the load/store must be left
// untouched for the unstructured gather/scatter lowering, which preserves
// the mask. Treating the comparison as always-true (previous behavior)
// drops the mask and emits a full-range DMA that reads below the base
// pointer.

module {
  tt.func @masked_shifted_sge_load(
    %arg0: !tt.ptr<f16>,
    %arg1: !tt.ptr<f16>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf16>
    %c0 = arith.constant dense<0> : tensor<256xi32>
    %c128 = arith.constant dense<128> : tensor<256xi32>
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %idx = arith.subi %offs, %c128 : tensor<256xi32>
    %mask = arith.cmpi sge, %idx, %c0 : tensor<256xi32>
    %ldptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %ldptr_1 = tt.addptr %ldptr, %idx : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %v = tt.load %ldptr_1, %mask, %cst : tensor<256x!tt.ptr<f16>>
    %stptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %stptr_1 = tt.addptr %stptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %stptr_1, %v, %mask : tensor<256x!tt.ptr<f16>>
    tt.return
  }

  // Control: `offs >= 0` has lower bound 0, still expressible as the full
  // prefix. The load must still be rewritten to tts.load with mask dims 256.
  tt.func @masked_plain_sge_load(
    %arg0: !tt.ptr<f16>,
    %arg1: !tt.ptr<f16>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf16>
    %c0 = arith.constant dense<0> : tensor<256xi32>
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %mask = arith.cmpi sge, %offs, %c0 : tensor<256xi32>
    %ldptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %ldptr_1 = tt.addptr %ldptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %v = tt.load %ldptr_1, %mask, %cst : tensor<256x!tt.ptr<f16>>
    %stptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %stptr_1 = tt.addptr %stptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %stptr_1, %v, %mask : tensor<256x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: tt.func @masked_shifted_sge_load
// CHECK: tt.load
// CHECK-NOT: tts.load
// CHECK-LABEL: tt.func @masked_plain_sge_load
// CHECK: "tts.load"
// CHECK-SAME: static_mask_dims = array<i64: 256>
