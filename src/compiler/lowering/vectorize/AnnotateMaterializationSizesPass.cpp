/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "compiler/lowering/vectorize/VectorUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cmath>
#include <cstddef>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/OpDefinition.h>

#include <memory>
#include <mlir/IR/Types.h>
#include <vector>

#include <llvm/ADT/STLExtras.h>

#include <spdlog/spdlog.h>
#include <util/ErrorHandler.h>

using namespace mlir;

namespace {

//-----------------------------------------------------------------
// Helper
//-----------------------------------------------------------------

size_t elementTypeToBits(mlir::Type eTy) {
    if (eTy.isF64())
        return 64;
    else if (eTy.isF32())
        return 32;
    else if (eTy.isSignedInteger(8))
        return 8;
    else if (eTy.isSignedInteger(32))
        return 32;
    else if (eTy.isSignedInteger(64))
        return 64;
    else if (eTy.isUnsignedInteger(8))
        return 8;
    else if (eTy.isUnsignedInteger(32))
        return 32;
    else if (eTy.isUnsignedInteger(64))
        return 64;
    else if (eTy.isSignlessInteger(1))
        return 1;

    return 0;
}

size_t estimateSize (daphne::MatrixType m) {

    auto eTySz = elementTypeToBits(m.getElementType());
    auto numCols = m.getNumCols();
    auto numRows = m.getNumRows();

    return eTySz * numRows * numCols;

}

//-----------------------------------------------------------------
// Class
//-----------------------------------------------------------------

struct AnnotateMaterializationSizesPass : public PassWrapper<AnnotateMaterializationSizesPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final;
};
} // namespace

void AnnotateMaterializationSizesPass::runOnOperation() {
    auto func = getOperation();

    func->walk([&](daphne::Vectorizable op) {

        bool isValidOp = false;
        for (auto opType : op->getOperandTypes()) {
            if (!opType.isIntOrIndexOrFloat() && !llvm::isa<daphne::StringType>(opType)) {
                isValidOp = true;
                break;
            }
        }
        if (isValidOp) {
            //estimation
            auto type = op->getResultTypes()[0];

            mlir::Builder builder(&getContext());
            if (auto m = type.dyn_cast<daphne::MatrixType>()) {
                mlir::IntegerAttr a = builder.getI64IntegerAttr(estimateSize(m));
                op->setAttr("M_SIZE", a);
            } else if (type.isIntOrIndexOrFloat() && !llvm::isa<daphne::StringType>(type)){
                mlir::IntegerAttr a = builder.getI64IntegerAttr(elementTypeToBits(type));
                op->setAttr("M_SIZE", a);
            } else {
                return;
            }

        }
    });

    return;
}

std::unique_ptr<Pass> daphne::createAnnotateMaterializationSizesPass() {
    return std::make_unique<AnnotateMaterializationSizesPass>();
}