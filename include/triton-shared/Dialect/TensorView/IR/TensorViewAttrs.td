//===-- TensorViewAttrs.td - TensorView attributes ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_ATTRS
#define TENSORVIEW_ATTRS

include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.td"

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Encoding attributes
//===----------------------------------------------------------------------===//

class TensorView_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TensorView_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let cppNamespace = "::mlir::triton::tv";
}

// #tv.partition_view<tile=[...], dim_map=[...]>
def TV_PartitionViewAttr : TensorView_Attr<"PartitionView", "partition_view"> {
  let summary = "Partition (tiling) view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$dimMap);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$tile), [{
      ::llvm::SmallVector<int64_t> dimMap;
      for (int64_t i = 0, e = tile.size(); i < e; ++i) dimMap.push_back(i);
      return $_get($_ctxt, tile, dimMap);
    }]>,
  ];
}

// #tv.strided_view<tile=[...], dim_map=[...], traversal_strides=[...]>
def TV_StridedViewAttr : TensorView_Attr<"StridedView", "strided_view"> {
  let summary = "Strided (partition + inter-tile stride) view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$dimMap,
    ArrayRefParameter<"int64_t">:$traversalStrides);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$tile,
                     "::llvm::ArrayRef<int64_t>":$traversalStrides), [{
      ::llvm::SmallVector<int64_t> dimMap;
      for (int64_t i = 0, e = tile.size(); i < e; ++i) dimMap.push_back(i);
      return $_get($_ctxt, tile, dimMap, traversalStrides);
    }]>,
  ];
}

// #tv.gather_scatter_view<tile=[...], sparse_dim=[...]>
def TV_GatherScatterViewAttr : TensorView_Attr<"GatherScatterView", "gather_scatter_view"> {
  let summary = "Gather/scatter view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$sparseDim);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // TENSORVIEW_ATTRS
