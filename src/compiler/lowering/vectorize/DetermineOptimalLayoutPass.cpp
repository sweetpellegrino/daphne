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
#include <mlir/IR/OpDefinition.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/STLExtras.h>

#include <spdlog/spdlog.h>
#include <util/ErrorHandler.h>

using namespace mlir;

namespace {

//-----------------------------------------------------------------
// Class
//-----------------------------------------------------------------

struct DetermineOptimalLayoutPass
    : public PassWrapper<DetermineOptimalLayoutPass, OperationPass<func::FuncOp>> {

    void runOnOperation() final;

    const DaphneUserConfig& userConfig;
    explicit DetermineOptimalLayoutPass(const DaphneUserConfig& cfg) : userConfig(cfg) {}

};

//-----------------------------------------------------------------
// Helper
//-----------------------------------------------------------------

bool hasMatrixAsInput(Operation *op) {
  for (mlir::Value operand : op->getOperands())
    if (operand.getType().isa<daphne::MatrixType>()) 
        return true;
  return false;
}

bool hasMatrixAsOutput(Operation *op) {
  for (mlir::Value operand : op->getResults())
    if (operand.getType().isa<daphne::MatrixType>()) 
        return true;
  return false;
}

} // namespace

void DetermineOptimalLayoutPass::runOnOperation() {

    auto funcOp = getOperation();
    funcOp->walk([&](mlir::Operation* op) {

        bool matrixInput = hasMatrixAsInput(op);
        bool matrixOutput = hasMatrixAsOutput(op);

        if(!matrixOutput)
            return;

        if(llvm::isa<daphne::VectorizedPipelineOp>(op))
            return;

        //llvm::SmallVector<Attribute> optimalLayouts(op->getNumResults());
        mlir::Attribute optimalLayout;
        if (!matrixInput) {
            optimalLayout = BoolAttr::get(op->getContext(), userConfig.isRowMajor);
            /*for (size_t i = 0; i < op->getNumResults(); ++i) {
                optimalLayouts[i] = mlir::BoolAttr::get(op->getContext(), userConfig.isRowMajor);
            }*/
        } else {

            if (llvm::isa<daphne::TransposeOp>(op)) {
                auto attr = op->getOperand(0).getDefiningOp()->getAttrOfType<mlir::BoolAttr>("shouldBeRowMajor");
                optimalLayout = mlir::BoolAttr::get(op->getContext(), !attr.getValue());
                //optimalLayouts[0] = mlir::BoolAttr::get(op->getContext(), !userConfig.isRowMajor);
            }
            else {
                optimalLayout = op->getOperand(0).getDefiningOp()->getAttrOfType<mlir::BoolAttr>("shouldBeRowMajor");
            }
        }
        //op->setAttr("shouldBeRowMajor", mlir::ArrayAttr::get(op->getContext(), optimalLayouts));
        op->setAttr("shouldBeRowMajor", optimalLayout);
    });

    return;
}

std::unique_ptr<Pass> daphne::createDetermineOptimalLayoutPass(const DaphneUserConfig& cfg) {
    return std::make_unique<DetermineOptimalLayoutPass>(cfg);
}