// RUN: triton-shared-opt --tensor-view-lowering %s | FileCheck %s

module {
  tt.func public @view_load(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
    %view = tv.make_partition_view %base_view : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>
    %result = tv.view_load %view[%index] : !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>, index -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @view_load(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDEX:%.*]]: index) -> tensor<4xf32> {
  // CHECK-DAG: [[PADDING:%.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: [[ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[ORIGIN]], [[C0]] : index
  // CHECK: [[END:%.*]] = arith.addi [[ORIGIN]], [[C4]] : index
  // CHECK: [[WITHIN_EXTENT:%.*]] = arith.cmpi sle, [[END]], [[SIZE]] : index
  // CHECK: [[IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[WITHIN_EXTENT]] : i1
  // CHECK: [[RESULT:%.*]] = scf.if [[IN_BOUNDS]] -> (tensor<4xf32>) {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<4xf32>
  // CHECK: [[TILE_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[TILE_OFFSET:%.*]] = arith.muli [[TILE_ORIGIN]], [[STRIDE]] : index
  // CHECK: [[GM_TILE:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[TILE_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  // CHECK: memref.copy [[GM_TILE]], [[BUFFER]] : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32>
  // CHECK: [[TENSOR:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<4xf32> to tensor<4xf32>
  // CHECK: scf.yield [[TENSOR]] : tensor<4xf32>
  // CHECK: } else {
  // CHECK: [[EMPTY:%.*]] = tensor.empty() : tensor<4xf32>
  // CHECK: [[FALLBACK:%.*]] = scf.for [[IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args([[ACC:%.*]] = [[EMPTY]]) -> (tensor<4xf32>) {
  // CHECK: [[FALLBACK_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[COORDINATE:%.*]] = arith.addi [[FALLBACK_ORIGIN]], [[IV]] : index
  // CHECK: [[ELEMENT_NON_NEGATIVE:%.*]] = arith.cmpi sge, [[COORDINATE]], [[C0]] : index
  // CHECK: [[ELEMENT_BELOW_EXTENT:%.*]] = arith.cmpi slt, [[COORDINATE]], [[SIZE]] : index
  // CHECK: [[ELEMENT_IN_BOUNDS:%.*]] = arith.andi [[ELEMENT_NON_NEGATIVE]], [[ELEMENT_BELOW_EXTENT]] : i1
  // CHECK: [[ELEMENT_OFFSET:%.*]] = arith.muli [[COORDINATE]], [[STRIDE]] : index
  // CHECK: [[ELEMENT:%.*]] = scf.if [[ELEMENT_IN_BOUNDS]] -> (f32) {
  // CHECK: [[GM_ELEMENT:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[ELEMENT_OFFSET]]{{.}}, sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK: [[LOADED:%.*]] = memref.load [[GM_ELEMENT]][[[C0]]] : memref<1xf32, strided<[1], offset: ?>>
  // CHECK: scf.yield [[LOADED]] : f32
  // CHECK: } else {
  // CHECK: scf.yield [[PADDING]] : f32
  // CHECK: [[INSERTED:%.*]] = tensor.insert [[ELEMENT]] into [[ACC]][[[IV]]] : tensor<4xf32>
  // CHECK: scf.yield [[INSERTED]] : tensor<4xf32>
  // CHECK: scf.yield [[FALLBACK]] : tensor<4xf32>
  // CHECK: tt.return [[RESULT]] : tensor<4xf32>

  tt.func public @view_store(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index, %value: tensor<4xf32>) {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
    %view = tv.make_strided_view %base_view : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.strided_view<tile = [4], dim_map = [0], traversal_strides = [8], padding_value = inf>>
    tv.view_store %view[%index], %value : !tv.tensor_view<?xf32, strides=[?], #tv.strided_view<tile = [4], dim_map = [0], traversal_strides = [8], padding_value = inf>>, tensor<4xf32>, index
    tt.return
  }

  // CHECK-LABEL: tt.func public @view_store(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDEX:%.*]]: index, [[VALUE:%.*]]: tensor<4xf32>) {
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C8:%.*]] = arith.constant 8 : index
  // CHECK: [[ORIGIN:%.*]] = arith.muli [[INDEX]], [[C8]] : index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[ORIGIN]], [[C0]] : index
  // CHECK: [[END:%.*]] = arith.addi [[ORIGIN]], [[C4]] : index
  // CHECK: [[WITHIN_EXTENT:%.*]] = arith.cmpi sle, [[END]], [[SIZE]] : index
  // CHECK: [[IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[WITHIN_EXTENT]] : i1
  // CHECK: scf.if [[IN_BOUNDS]] {
  // CHECK: [[TILE_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C8]] : index
  // CHECK: [[TILE_OFFSET:%.*]] = arith.muli [[TILE_ORIGIN]], [[STRIDE]] : index
  // CHECK: [[GM_TILE:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[TILE_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  // CHECK: bufferization.materialize_in_destination [[VALUE]] in writable [[GM_TILE]] : (tensor<4xf32>, memref<4xf32, strided<[?], offset: ?>>) -> ()
  // CHECK: } else {
  // CHECK: scf.for [[IV:%.*]] = [[C0]] to [[C4]] step [[C1]] {
  // CHECK: [[FALLBACK_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C8]] : index
  // CHECK: [[COORDINATE:%.*]] = arith.addi [[FALLBACK_ORIGIN]], [[IV]] : index
  // CHECK: [[ELEMENT_NON_NEGATIVE:%.*]] = arith.cmpi sge, [[COORDINATE]], [[C0]] : index
  // CHECK: [[ELEMENT_BELOW_EXTENT:%.*]] = arith.cmpi slt, [[COORDINATE]], [[SIZE]] : index
  // CHECK: [[ELEMENT_IN_BOUNDS:%.*]] = arith.andi [[ELEMENT_NON_NEGATIVE]], [[ELEMENT_BELOW_EXTENT]] : i1
  // CHECK: [[ELEMENT_OFFSET:%.*]] = arith.muli [[COORDINATE]], [[STRIDE]] : index
  // CHECK: scf.if [[ELEMENT_IN_BOUNDS]] {
  // CHECK: [[ELEMENT:%.*]] = tensor.extract [[VALUE]][[[IV]]] : tensor<4xf32>
  // CHECK: [[GM_ELEMENT:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[ELEMENT_OFFSET]]{{.}}, sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK: [[EMPTY:%.*]] = tensor.empty() : tensor<1xf32>
  // CHECK: [[INSERTED:%.*]] = tensor.insert [[ELEMENT]] into [[EMPTY]][[[C0]]] : tensor<1xf32>
  // CHECK: bufferization.materialize_in_destination [[INSERTED]] in writable [[GM_ELEMENT]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
  // CHECK: tt.return

  tt.func public @gather_padding(%base: !tv.ptr<f32>, %size: index, %stride: index, %indices: tensor<4xi32>) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
    %view = tv.make_gather_scatter_view %base_view : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.gather_scatter_view<tile = [4], sparse_dim = [0], padding_value = nan>>
    %result = tv.view_load %view[%indices] : !tv.tensor_view<?xf32, strides=[?], #tv.gather_scatter_view<tile = [4], sparse_dim = [0], padding_value = nan>>, tensor<4xi32> -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @gather_padding(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDICES:%.*]]: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK-DAG: [[NAN:%.*]] = arith.constant 0x7FC00000 : f32
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[TRUE:%.*]] = arith.constant true
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: [[ALL_IN_BOUNDS:%.*]] = scf.for [[CHECK_IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args([[VALID:%.*]] = [[TRUE]]) -> (i1) {
  // CHECK: [[RAW_INDEX:%.*]] = tensor.extract [[INDICES]][[[CHECK_IV]]] : tensor<4xi32>
  // CHECK: [[INDEX:%.*]] = arith.index_cast [[RAW_INDEX]] : i32 to index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[INDEX]], [[C0]] : index
  // CHECK: [[BELOW_EXTENT:%.*]] = arith.cmpi slt, [[INDEX]], [[SIZE]] : index
  // CHECK: [[INDEX_IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[BELOW_EXTENT]] : i1
  // CHECK: [[NEXT_VALID:%.*]] = arith.andi [[VALID]], [[INDEX_IN_BOUNDS]] : i1
  // CHECK: scf.yield [[NEXT_VALID]] : i1
  // CHECK: [[RESULT:%.*]] = scf.if [[ALL_IN_BOUNDS]] -> (tensor<4xf32>) {
  // CHECK: [[EMPTY:%.*]] = tensor.empty() : tensor<4xf32>
  // CHECK: [[GATHERED:%.*]] = scf.for [[IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args([[ACC:%.*]] = [[EMPTY]]) -> (tensor<4xf32>) {
  // CHECK: [[RAW_COORDINATE:%.*]] = tensor.extract [[INDICES]][[[IV]]] : tensor<4xi32>
  // CHECK: [[COORDINATE:%.*]] = arith.index_cast [[RAW_COORDINATE]] : i32 to index
  // CHECK: [[ELEMENT_NON_NEGATIVE:%.*]] = arith.cmpi sge, [[COORDINATE]], [[C0]] : index
  // CHECK: [[ELEMENT_BELOW_EXTENT:%.*]] = arith.cmpi slt, [[COORDINATE]], [[SIZE]] : index
  // CHECK: [[ELEMENT_IN_BOUNDS:%.*]] = arith.andi [[ELEMENT_NON_NEGATIVE]], [[ELEMENT_BELOW_EXTENT]] : i1
  // CHECK: [[ELEMENT_OFFSET:%.*]] = arith.muli [[COORDINATE]], [[STRIDE]] : index
  // CHECK: [[ELEMENT:%.*]] = scf.if [[ELEMENT_IN_BOUNDS]] -> (f32) {
  // CHECK: [[GM_ELEMENT:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[ELEMENT_OFFSET]]{{.}}, sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK: [[LOADED:%.*]] = memref.load [[GM_ELEMENT]][[[C0]]] : memref<1xf32, strided<[1], offset: ?>>
  // CHECK: scf.yield [[LOADED]] : f32
  // CHECK: } else {
  // CHECK: scf.yield [[NAN]] : f32
  // CHECK: [[INSERTED:%.*]] = tensor.insert [[ELEMENT]] into [[ACC]][[[IV]]] : tensor<4xf32>
  // CHECK: scf.yield [[INSERTED]] : tensor<4xf32>
  // CHECK: scf.yield [[GATHERED]] : tensor<4xf32>
  // CHECK: } else {
  // CHECK: [[FALLBACK_EMPTY:%.*]] = tensor.empty() : tensor<4xf32>
  // CHECK: [[FALLBACK:%.*]] = scf.for [[FALLBACK_IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args([[FALLBACK_ACC:%.*]] = [[FALLBACK_EMPTY]]) -> (tensor<4xf32>) {
  // CHECK: [[FALLBACK_RAW_COORDINATE:%.*]] = tensor.extract [[INDICES]][[[FALLBACK_IV]]] : tensor<4xi32>
  // CHECK: [[FALLBACK_COORDINATE:%.*]] = arith.index_cast [[FALLBACK_RAW_COORDINATE]] : i32 to index
  // CHECK: [[FALLBACK_OFFSET:%.*]] = arith.muli [[FALLBACK_COORDINATE]], [[STRIDE]] : index
  // CHECK: [[FALLBACK_ELEMENT:%.*]] = scf.if {{%.*}} -> (f32) {
  // CHECK: [[FALLBACK_GM_ELEMENT:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[FALLBACK_OFFSET]]{{.}}, sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK: [[FALLBACK_LOADED:%.*]] = memref.load [[FALLBACK_GM_ELEMENT]][[[C0]]] : memref<1xf32, strided<[1], offset: ?>>
  // CHECK: scf.yield [[FALLBACK_LOADED]] : f32
  // CHECK: } else {
  // CHECK: scf.yield [[NAN]] : f32
  // CHECK: [[FALLBACK_INSERTED:%.*]] = tensor.insert [[FALLBACK_ELEMENT]] into [[FALLBACK_ACC]][[[FALLBACK_IV]]] : tensor<4xf32>
  // CHECK: scf.yield [[FALLBACK_INSERTED]] : tensor<4xf32>
  // CHECK: scf.yield [[FALLBACK]] : tensor<4xf32>
  // CHECK: tt.return [[RESULT]] : tensor<4xf32>

  tt.func public @negative_infinity_padding(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
    %view = tv.make_partition_view %base_view : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = -inf>>
    %result = tv.view_load %view[%index] : !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = -inf>>, index -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @negative_infinity_padding(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDEX:%.*]]: index) -> tensor<4xf32> {
  // CHECK-DAG: [[NEGATIVE_INFINITY:%.*]] = arith.constant 0xFF800000 : f32
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: [[ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[ORIGIN]], [[C0]] : index
  // CHECK: [[END:%.*]] = arith.addi [[ORIGIN]], [[C4]] : index
  // CHECK: [[WITHIN_EXTENT:%.*]] = arith.cmpi sle, [[END]], [[SIZE]] : index
  // CHECK: [[IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[WITHIN_EXTENT]] : i1
  // CHECK: [[RESULT:%.*]] = scf.if [[IN_BOUNDS]] -> (tensor<4xf32>) {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<4xf32>
  // CHECK: [[TILE_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[TILE_OFFSET:%.*]] = arith.muli [[TILE_ORIGIN]], [[STRIDE]] : index
  // CHECK: [[GM_TILE:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[TILE_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  // CHECK: memref.copy [[GM_TILE]], [[BUFFER]] : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32>
  // CHECK: [[TENSOR:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<4xf32> to tensor<4xf32>
  // CHECK: scf.yield [[TENSOR]] : tensor<4xf32>
  // CHECK: } else {
  // CHECK: scf.for [[IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args({{%.*}} = {{%.*}}) -> (tensor<4xf32>) {
  // CHECK: [[ELEMENT:%.*]] = scf.if {{%.*}} -> (f32) {
  // CHECK: memref.load
  // CHECK: } else {
  // CHECK: scf.yield [[NEGATIVE_INFINITY]] : f32
  // CHECK: tt.return [[RESULT]] : tensor<4xf32>

  tt.func public @integer_zero_padding(%base: !tv.ptr<i32>, %size: index, %stride: index, %index: index) -> tensor<4xi32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<i32> -> !tv.tensor_view<?xi32, strides=[?]>
    %view = tv.make_partition_view %base_view : !tv.tensor_view<?xi32, strides=[?]> -> !tv.tensor_view<?xi32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>
    %result = tv.view_load %view[%index] : !tv.tensor_view<?xi32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>, index -> tensor<4xi32>
    tt.return %result : tensor<4xi32>
  }

  // CHECK-LABEL: tt.func public @integer_zero_padding(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xi32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDEX:%.*]]: index) -> tensor<4xi32> {
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0 : i32
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: [[ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[ORIGIN]], [[C0]] : index
  // CHECK: [[END:%.*]] = arith.addi [[ORIGIN]], [[C4]] : index
  // CHECK: [[WITHIN_EXTENT:%.*]] = arith.cmpi sle, [[END]], [[SIZE]] : index
  // CHECK: [[IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[WITHIN_EXTENT]] : i1
  // CHECK: [[RESULT:%.*]] = scf.if [[IN_BOUNDS]] -> (tensor<4xi32>) {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<4xi32>
  // CHECK: [[TILE_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[TILE_OFFSET:%.*]] = arith.muli [[TILE_ORIGIN]], [[STRIDE]] : index
  // CHECK: [[GM_TILE:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[TILE_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xi32> to memref<4xi32, strided<[?], offset: ?>>
  // CHECK: memref.copy [[GM_TILE]], [[BUFFER]] : memref<4xi32, strided<[?], offset: ?>> to memref<4xi32>
  // CHECK: [[TENSOR:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<4xi32> to tensor<4xi32>
  // CHECK: scf.yield [[TENSOR]] : tensor<4xi32>
  // CHECK: } else {
  // CHECK: scf.for [[IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args({{%.*}} = {{%.*}}) -> (tensor<4xi32>) {
  // CHECK: [[ELEMENT:%.*]] = scf.if {{%.*}} -> (i32) {
  // CHECK: memref.load
  // CHECK: } else {
  // CHECK: scf.yield [[ZERO]] : i32
  // CHECK: tt.return [[RESULT]] : tensor<4xi32>
}
