/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_TRITON_TO_LINALG_INCUBATED_COMPAT_H
#define TRITON_ADAPTER_TRITON_TO_LINALG_INCUBATED_COMPAT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

// Keep these math ops on the tensor lowering path on Ascend. Their scalar
// forms are not reliably supported by the device compiler, including when
// computing pointer offsets. In particular, ceil/floor must not be replaced
// by an integer round-trip: that restricts the input range and loses -0.
inline bool hasOnlyVectorizedDeviceImpl(Operation *op) {
  return isa<math::CeilOp, math::FloorOp, math::ExpOp, math::Exp2Op,
             math::Log2Op, math::SinOp, math::CosOp, math::ErfOp>(op);
}

// True when `v` feeds a tt.addptr offset through scalar or tensor arith/math
// and tt.splat. A value may also have data uses; one offset use is sufficient.
inline bool scalarValueFeedsPointerOffset(Value v) {
  llvm::SmallVector<Value> worklist{v};
  llvm::SmallPtrSet<Operation *, 8> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (auto addPtr = dyn_cast<triton::AddPtrOp>(user)) {
        if (addPtr.getOffset() == cur)
          return true;
        continue;
      }
      if (!seen.insert(user).second || user->getNumResults() != 1)
        continue;
      if (!isa<triton::SplatOp>(user) &&
          !isa<arith::ArithDialect, math::MathDialect>(user->getDialect()))
        continue;
      Value res = user->getResult(0);
      Type type = res.getType();
      if (auto tensorType = dyn_cast<RankedTensorType>(type))
        type = tensorType.getElementType();
      if (!type.isIntOrIndexOrFloat())
        continue;
      worklist.push_back(res);
    }
  }
  return false;
}

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_LINALG_INCUBATED_COMPAT_H
