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

#ifndef TRITON_INCUBATED_CONTROL_FLOW_ATTRS_H
#define TRITON_INCUBATED_CONTROL_FLOW_ATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::controlflow {

/// Handoff marker for SCF loops whose pointer slots have already been expanded
/// into policy-owned descriptor components. TritonToLinalg consumes and
/// removes it instead of applying its legacy pointer-loop decomposition a
/// second time.
inline constexpr llvm::StringLiteral kPointerDescriptorBoundaryAttr =
    "PointerDescriptorBoundary";

/// Handoff marker for a pointer rebuilt at a CFO control-flow boundary. The
/// defining operation's operands are the complete descriptor roots, including
/// invariant components omitted from the replacement SCF signature.
inline constexpr llvm::StringLiteral kPointerDescriptorRebuildAttr =
    "PointerDescriptorRebuild";

/// Optional companion on a PointerDescriptorRebuild root. The `strided_1d`
/// form means the tensor-pointer offset operand is a proven rank-1 affine
/// expression rather than an opaque complete-offset carrier.
inline constexpr llvm::StringLiteral kPointerDescriptorOffsetFormAttr =
    "PointerDescriptorOffsetForm";
inline constexpr llvm::StringLiteral kStrided1DOffsetForm = "strided_1d";

/// Dense per-axis classification attached to a descriptor rebuild. A value
/// of one means the complete contribution of that logical axis is represented
/// by its scalar stride; zero means the axis is carried by the opaque complete
/// contribution tensor.
inline constexpr llvm::StringLiteral kPointerDescriptorStructuredAxesAttr =
    "PointerDescriptorStructuredAxes";

} // namespace mlir::triton::controlflow

#endif // TRITON_INCUBATED_CONTROL_FLOW_ATTRS_H
