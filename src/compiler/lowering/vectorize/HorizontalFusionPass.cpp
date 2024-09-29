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
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstddef>
#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/MathExtras.h>
#include <map>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <llvm/ADT/STLExtras.h>
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <spdlog/spdlog.h>
#include <util/ErrorHandler.h>

using namespace mlir;

namespace
{

    struct HorizontalFusionPass : public PassWrapper<HorizontalFusionPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
    };

}

//-----------------------------------------------------------------
// Consumer <- Producer -> Consumer
//-----------------------------------------------------------------

//TODO: change me
//Based on the operand, check if the operand is also used from another operation
//If yes, these are potentially candidates for horizontal fusion
//Horizontal Fusion / Sibling Fusion:
//
//          producer
//         /        \
//        opv       user
//
// => (opv, user)

void HorizontalFusionPass::runOnOperation()
{   
    
    bool changed = true;
    while(changed) {
        changed = false;

        auto func = getOperation();

        std::vector<daphne::VectorizedPipelineOp> pipelineOps;
        func->walk([&](daphne::VectorizedPipelineOp op) {
            pipelineOps.emplace_back(op);
        });
        std::reverse(pipelineOps.begin(), pipelineOps.end()); 

        // Check for overlapping/intersection between pipeline arguments
        // Should work with bounding operands and operands between the pipelines.
        std::vector<PipelineOpPair> horizontalRelationships;
        for (auto it1 = pipelineOps.begin(); it1 != pipelineOps.end(); ++it1) {
            auto pipeOp1 = *it1;

            //pipeOp1->dump();
            //llvm::outs() << "\n";

            // Store defOps for the corresponding args
            llvm::SmallVector<mlir::Operation*> defOpsArgs;
            // Running over split size allows for only considering of relevant args. e.g. excl. outCols etc.
            for(size_t operandIx1 = 0; operandIx1 < pipeOp1.getSplits().size(); ++operandIx1) {
                auto operand1 = pipeOp1->getOperand(operandIx1);
                if (auto defOp = operand1.getDefiningOp()) {
                    defOpsArgs.push_back(defOp);
                }
            }

            for (auto it2 = next(it1); it2 != pipelineOps.end(); ++it2) {
                auto pipeOp2 = *it2;

                //pipeOp2->dump();
                //llvm::outs() << "\n";

                for(size_t operandIx2 = 0; operandIx2 < pipeOp2.getSplits().size(); ++operandIx2) {
                    auto operand2 = pipeOp2->getOperand(operandIx2);

                    if (auto defOp = operand2.getDefiningOp()) {

                        auto fIt = std::find(defOpsArgs.begin(), defOpsArgs.end(), defOp);
                        if (fIt != defOpsArgs.end()) {
                            
                            size_t operandIx1 = std::distance(defOpsArgs.begin(), fIt);

                            if (pipeOp1.getSplits()[operandIx1] == pipeOp2.getSplits()[operandIx2] && 
                                pipeOp1.getSplits()[operandIx1].cast<daphne::VectorSplitAttr>().getValue() != daphne::VectorSplit::NONE) {
                                llvm::outs() << operandIx1 << ", " << operandIx2 << "\n";
                                //TODO: but what if there is some other sharing argument that is invalid?
                                horizontalRelationships.push_back({pipeOp1, pipeOp2});
                                break; //only one compatabile argument needs to be found
                            }
                        }
                    }
                }
            }
        }

        //After merging one of the pairs, if theoretically need to rerun the pass
        //and invalidate every other pair with one of these pipeOps.
        for(auto pipeOpPair : horizontalRelationships) {
        
            auto pipeOp1 = pipeOpPair.first;
            auto pipeOp2 = pipeOpPair.second;

            //currently no earlier check if pipelines are in different blocks
            //you can check with gnmf.daph
            if (pipeOp1->getBlock() != pipeOp2->getBlock())
                continue;

            if (VectorUtils::arePipelineOpsDependent(pipeOp1, pipeOp2))
                continue;

            /*llvm::outs() << "--------------------------------------------------------------------------------" << "\n";
            pipeOp1->dump();
            pipeOp2->dump();
            llvm::outs() << "\n";
            llvm::outs() << "--------------------------------------------------------------------------------" << "\n";*/

            mlir::Block* b1 = &pipeOp1.getBody().getBlocks().front();
            mlir::Block* b2 = &pipeOp2.getBody().getBlocks().front();

            auto vSplitAttrs = std::vector<mlir::Attribute>(pipeOp1.getSplits().begin(), pipeOp1.getSplits().end());
            auto vCombineAttrs = std::vector<mlir::Attribute>(pipeOp1.getCombines().begin(), pipeOp1.getCombines().end());
            auto results = std::vector<mlir::Value>(pipeOp1->getResults().begin(), pipeOp1->getResults().end());
            auto operands = std::vector<mlir::Value>(pipeOp1->getOperands().begin(), pipeOp1->getOperands().begin() + pipeOp1.getSplits().size());
            auto outRows = std::vector<mlir::Value>(pipeOp1.getOutRows().begin(), pipeOp1.getOutRows().end());
            auto outCols = std::vector<mlir::Value>(pipeOp1.getOutCols().begin(), pipeOp1.getOutCols().end());

            vSplitAttrs.insert(vSplitAttrs.end(), pipeOp2.getSplits().begin(), pipeOp2.getSplits().end());
            vCombineAttrs.insert(vCombineAttrs.end(), pipeOp2.getCombines().begin(), pipeOp2.getCombines().end());
            results.insert(results.end(), pipeOp2->getResults().begin(), pipeOp2->getResults().end());
            operands.insert(operands.end(), pipeOp2->getOperands().begin(), pipeOp2->getOperands().begin() + pipeOp2.getSplits().size());
            outRows.insert(outRows.end(), pipeOp2.getOutRows().begin(), pipeOp2.getOutRows().end());
            outCols.insert(outCols.end(), pipeOp2.getOutCols().begin(), pipeOp2.getOutCols().end());

            mlir::OpBuilder builder(func);
            auto loc = builder.getFusedLoc({pipeOp1.getLoc(), pipeOp2->getLoc()});
            auto pipelineOp = builder.create<mlir::daphne::VectorizedPipelineOp>(loc,
                mlir::ValueRange(results).getTypes(),
                operands,
                outRows,
                outCols,
                builder.getArrayAttr(vSplitAttrs),
                builder.getArrayAttr(vCombineAttrs),
                nullptr);
            mlir::Block *bodyBlock = builder.createBlock(&pipelineOp.getBody()); 

            auto operations = std::vector<mlir::Operation*>();
            auto _results = std::vector<mlir::Value>();
            while(!b1->empty()) {
                auto op = b1->begin();
            //for (auto &op : llvm::make_early_inc_range(pipeOp1.getBody().getBlocks().front())) {

                for(size_t i = 0; i < op->getNumOperands(); ++i) {
                    auto operand = op->getOperand(i);
                    if (llvm::isa<mlir::BlockArgument>(operand)) {
                        auto blockArgument = bodyBlock->addArgument(operand.getType(), builder.getUnknownLoc());
                        op->setOperand(i, blockArgument);
                    }
                }
                if (!llvm::isa<daphne::ReturnOp>(op)) {
                    op->moveBefore(bodyBlock, bodyBlock->end());
                }
                else {
                    for (auto operand : op->getOperands()) {
                        _results.push_back(operand);
                    }
                    //op->moveBefore(bodyBlock, bodyBlock->end());
                    op->erase();
                    break;
                }
            }

            while(!b2->empty()) {
                auto op = b2->begin();
            //for (auto &op : llvm::make_early_inc_range(pipeOp1.getBody().getBlocks().front())) {

                for(size_t i = 0; i < op->getNumOperands(); ++i) {
                    auto operand = op->getOperand(i);
                    if (llvm::isa<mlir::BlockArgument>(operand)) {
                        auto blockArgument = bodyBlock->addArgument(operand.getType(), builder.getUnknownLoc());
                        op->setOperand(i, blockArgument);
                    }
                }
                if (!llvm::isa<daphne::ReturnOp>(op)) {
                    op->moveBefore(bodyBlock, bodyBlock->end());
                }
                else {
                    for (auto operand : op->getOperands()) {
                        _results.push_back(operand);
                    }
                    //op->moveBefore(bodyBlock, bodyBlock->end());
                    op->erase();
                    break;
                }
            }

            builder.setInsertionPointToEnd(bodyBlock);
            builder.create<mlir::daphne::ReturnOp>(loc, _results);

            for (size_t i = 0; i < results.size(); ++i) {
                results.at(i).replaceAllUsesWith(pipelineOp.getResult(i));
            }

            /*pipelineOp->dump();
            llvm::outs() << "--------------------------------------------------------------------------------" << "\n";*/

            // Is this sufficient?
            pipelineOp->moveAfter(pipeOp1);

            pipeOp1->erase();
            pipeOp2->erase();
            changed = true;
        }
    }

    return;
}


std::unique_ptr<Pass> daphne::createHorizontalFusionPass() {
    return std::make_unique<HorizontalFusionPass>();
}