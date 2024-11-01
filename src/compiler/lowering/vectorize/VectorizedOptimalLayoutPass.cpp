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

struct VectorizedOptimalLayoutPass
    : public PassWrapper<VectorizedOptimalLayoutPass, OperationPass<func::FuncOp>> {

    void runOnOperation() final;

    const DaphneUserConfig& userConfig;
    explicit VectorizedOptimalLayoutPass(const DaphneUserConfig& cfg) : userConfig(cfg) {}

};
} // namespace

void VectorizedOptimalLayoutPass::runOnOperation() {

    auto funcOp = getOperation();
    funcOp->walk([&](daphne::VectorizedPipelineOp op) {

        SmallVector<Attribute> optimalLayouts(op->getNumResults());

        // daphne::VectorizedPipelineOp vectOp = llvm::dyn_cast<daphne::VectorizedPipelineOp>(op);
        mlir::Operation* result = &op.getBody().getBlocks().front().getOperations().back();

        for (size_t i = 0; i < result->getNumOperands(); ++i) {
            //auto attr = result->getOperand(i).getDefiningOp()->getAttrOfType<mlir::BoolAttr>("shouldBeRowMajor");
            optimalLayouts[i] = result->getOperand(i).getDefiningOp()->getAttrOfType<mlir::BoolAttr>("shouldBeRowMajor");
        }
        op->setAttr("shouldBeRowMajor", mlir::ArrayAttr::get(op->getContext(), optimalLayouts));
    }); 

    return;
}

std::unique_ptr<Pass> daphne::createVectorizedOptimalLayoutPass(const DaphneUserConfig& cfg) {
    return std::make_unique<VectorizedOptimalLayoutPass>(cfg);
}