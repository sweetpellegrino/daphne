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
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cmath>
#include <mlir/IR/OpDefinition.h>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <unordered_set>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include <map>

#include <llvm/ADT/STLExtras.h>
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <spdlog/spdlog.h>
#include <util/ErrorHandler.h>

using namespace mlir;

namespace
{

    //-----------------------------------------------------------------
    // Class
    //-----------------------------------------------------------------

    struct DrawPipelineOpsPass : public PassWrapper<DrawPipelineOpsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
        const std::string filename;

        explicit DrawPipelineOpsPass(const std::string filename) : filename(filename) {}

    }; 
}

    
void DrawPipelineOpsPass::runOnOperation()
{
    auto func = getOperation();

    std::vector<daphne::VectorizedPipelineOp> pipelineOps;
    func->walk([&](daphne::VectorizedPipelineOp op) {
        pipelineOps.emplace_back(op);
    });

    VectorUtils::DEBUG::drawPipelineOps(pipelineOps, filename);

    return;
}


std::unique_ptr<Pass> daphne::createDrawPipelineOpsPass(const std::string filename) {
    return std::make_unique<DrawPipelineOpsPass>(filename);
}