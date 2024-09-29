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

    llvm::outs() << "HorizontalFusionPass" << "\n";

    auto func = getOperation();

    llvm::outs() << "TEST" << "\n";

    /*func->print(llvm::outs());
    llvm::outs() << "\n";'*/

    std::vector<daphne::VectorizedPipelineOp> pipelineOps;
    func->walk([&](daphne::VectorizedPipelineOp op) {
        pipelineOps.emplace_back(op);
    });
    std::reverse(pipelineOps.begin(), pipelineOps.end()); 

    // Check for overlapping/intersection between pipeline arguments
    std::vector<PipelineOpPair> horizontalRelationships;
    for (auto it1 = pipelineOps.begin(); it1 != pipelineOps.end(); ++it1) {
        auto pipeOp1 = *it1;

        pipeOp1->dump();
        llvm::outs() << "\n";

        // Store defOps for the corresponding args
        llvm::SmallVector<mlir::Operation*> defOpsArgs;
        // Running over split size allows for only considering of relevant args. e.g. excl. context etc.
        for(size_t operandIx1 = 0; operandIx1 < pipeOp1.getSplits().size(); ++operandIx1) {
            auto operand1 = pipeOp1->getOperand(operandIx1);
            if (auto defOp = operand1.getDefiningOp()) {
                defOpsArgs.push_back(defOp);
            }
        }

        for (auto it2 = next(it1); it2 != pipelineOps.end(); ++it2) {
            auto pipeOp2 = *it2;

            pipeOp2->dump();
            llvm::outs() << "\n";

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

    for(auto pipeOpPair : horizontalRelationships) {

        auto pipeOp1 = pipeOpPair.first;
        auto pipeOp2 = pipeOpPair.second;

        //currently no earlier check if pipelines are in different blocks
        //you can check with gnmf.daph
        if (pipeOp1->getBlock() != pipeOp2->getBlock())
            continue;

        if (VectorUtils::arePipelineOpsDependent(pipeOp1, pipeOp2))
            continue;

        llvm::outs() << "--------------------------------------------------------------------------------" << "\n";
        pipeOp1->dump();
        pipeOp2->dump();
        llvm::outs() << "\n";
        llvm::outs() << "--------------------------------------------------------------------------------" << "\n";

        /*mlir::Block* b1 = &pipeOp1.getBody().getBlocks().front();
        mlir::Block* b2 = &pipeOp2.getBody().getBlocks().front();
        Pipeline merged;
        for (auto it = b2->begin(), ie = b2->end(); it != ie; ++it) {
            mlir::Operation* op = &(*it);
            merged.push_back(op);
        }
        for (auto it = b1->begin(), ie = b1->end(); it != ie; ++it) {
            mlir::Operation* op = &(*it);
            merged.push_back(op);
        }

        std::map<mlir::Operation*, VectorIndex> decisionIxs;
        for (auto op : merged) {
            decisionIxs.insert({op, 0});
        }
        
        VectorUtils::createVectorizedPipelineOps(func, {merged}, decisionIxs);*/

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
        auto loc = pipeOp2->getLoc();
        auto pipelineOp = builder.create<mlir::daphne::VectorizedPipelineOp>(loc,
            mlir::ValueRange(results).getTypes(),
            operands,
            outRows,
            outCols,
            builder.getArrayAttr(vSplitAttrs),
            builder.getArrayAttr(vCombineAttrs),
            nullptr);
        mlir::Block *bodyBlock = builder.createBlock(&pipelineOp.getBody()); 

        /*for (auto operand : operands) {
            bodyBlock->addArgument(operand.getType(), builder.getUnknownLoc());
        }*/
        auto operations = std::vector<mlir::Operation*>();
        for (auto &op : pipeOp1.getBody().getBlocks().front()) {

            op.dump();
            llvm::outs() << "\n";

            for(size_t i = 0; i < op.getNumOperands(); ++i) {
                auto operand = op.getOperand(i);
                if (llvm::isa<mlir::BlockArgument>(operand)) {
                    auto blockArgument = bodyBlock->addArgument(operand.getType(), builder.getUnknownLoc());
                    op.setOperand(i, blockArgument);
                }
            }
            if (!llvm::isa<daphne::ReturnOp>(op)) {
                op.moveBefore(bodyBlock, bodyBlock->end());
            }
            else {

            }
        }
        for (auto &op : pipeOp2.getBody().getBlocks().front()) {

            op.dump();
            llvm::outs() << "\n";

            for(size_t i = 0; i < op.getNumOperands(); ++i) {
                auto operand = op.getOperand(i);
                if (llvm::isa<mlir::BlockArgument>(operand)) {
                    auto blockArgument = bodyBlock->addArgument(operand.getType(), builder.getUnknownLoc());
                    op.setOperand(i, blockArgument);
                }
            }
            if (!llvm::isa<daphne::ReturnOp>(op)) {
                op.moveBefore(bodyBlock, bodyBlock->end());
            }
            else {

            }
        }


        /*mlir::Block* b1 = &pipeOp1.getBody().getBlocks().front();
        mlir::Block* b2 = &pipeOp2.getBody().getBlocks().front();

        Pipeline merged;
        for (auto it = b2->begin(), ie = b2->end(); it != ie; ++it) {
            mlir::Operation* op = &(*it);
            merged.push_back(op);
        }
        for (auto it = b1->begin(), ie = b1->end(); it != ie; ++it) {
            mlir::Operation* op = &(*it);
            merged.push_back(op);
        }

        std::map<mlir::Operation*, VectorIndex> decisionIxs;
        for (auto op : merged) {
            decisionIxs.insert({op, 0});
        }

        auto valueIsPartOfPipeline = [&](mlir::Value operand) {
            return llvm::any_of(merged, [&](mlir::Operation* lv) { return lv == operand.getDefiningOp(); });
        };

        //pipeOp1->replaceAllUsesWith(ValuesT &&values)

        for (auto operand : operands) {
            bodyBlock->addArgument(operand.getType(), builder.getUnknownLoc());
        }

        auto argsIx = 0u;
        auto resultsIx = 0u;
        //for every op in pipeline
        for(auto vIt = merged.begin(); vIt != merged.end(); ++vIt) {
            auto v = *vIt;
            auto numOperands = v->getNumOperands();
            auto numResults = v->getNumResults();

            //move v before end of block
            if (!llvm::isa<daphne::ReturnOp>(v)) {
                v->moveBefore(bodyBlock, bodyBlock->end());
            }

            //set operands to arguments of body block, if defOp is not part of the pipeline
            for(auto i = 0u; i < numOperands; ++i) {
                if(!valueIsPartOfPipeline(v->getOperand(i))) {
                    v->setOperand(i, bodyBlock->getArgument(argsIx++));
                }
            }

            llvm::outs() << "--------------------------------------------------------------------------------" << "\n";
            pipelineOp->dump();

            auto pipelineReplaceResults = pipelineOp->getResults().drop_front(resultsIx).take_front(numResults);
            resultsIx += numResults;
            for (auto z : llvm::zip(v->getResults(), pipelineReplaceResults)) {
                auto old = std::get<0>(z);
                auto replacement = std::get<1>(z);

                // TODO: switch to type based size inference instead
                // FIXME: if output is dynamic sized, we can't do this
                // replace `NumRowOp` and `NumColOp`s for output size inference
                for(auto& use: old.getUses()) {
                    auto* op = use.getOwner();
                    if(auto nrowOp = llvm::dyn_cast<mlir::daphne::NumRowsOp>(op)) {
                        nrowOp.replaceAllUsesWith(pipelineOp.getOutRows()[replacement.getResultNumber()]);
                        nrowOp.erase();
                    }
                    if(auto ncolOp = llvm::dyn_cast<mlir::daphne::NumColsOp>(op)) {
                        ncolOp.replaceAllUsesWith(pipelineOp.getOutCols()[replacement.getResultNumber()]);
                        ncolOp.erase();
                    }
                }
                // Replace only if not used by pipeline op
                old.replaceUsesWithIf(
                    replacement, [&](mlir::OpOperand &opOperand) {
                        return llvm::count(merged, opOperand.getOwner()) == 0;
                    });
                }
        }
        bodyBlock->walk([](mlir::Operation* op) {
            for(auto resVal: op->getResults()) {
                if(auto ty = resVal.getType().dyn_cast<mlir::daphne::MatrixType>()) {
                    resVal.setType(ty.withShape(-1, -1));
                }
            }
        });*/
        builder.setInsertionPointToEnd(bodyBlock);
        builder.create<mlir::daphne::ReturnOp>(loc, results);

        pipelineOp->dump();


        //pipeOp1->erase();
        //pipeOp2->erase();

        // Merge blocks and rest
        //auto loc = builder.getFusedLoc(locs);
        /*pipeOp1.getBody().getBlocks().front().dump();
        mlir::Block* b1 = &pipeOp1.getBody().getBlocks().front();
        mlir::Block* b2 = &pipeOp2.getBody().getBlocks().front();

        llvm::outs() << "--------------------------------------------------------------------------------" << "\n";
        pipeOp1.getBody().getBlocks().front().getOperations().splice(pipeOp1.getBody().getBlocks().front().end(), pipeOp2.getBody().getBlocks().front().getOperations());
        pipeOp1.getBody().getBlocks().front().dump();
        llvm::outs() << "--------------------------------------------------------------------------------" << "\n";
        llvm::SmallVector<mlir::Location> locs;
        for (size_t i = 0; i < pipeOp2.getBody().getBlocks().front().getArguments().size(); ++i) {
            locs.push_back(pipeOp1->getLoc());
        }
        pipeOp1.getBody().getBlocks().front().addArguments(pipeOp2.getBody().getBlocks().front().getArgumentTypes(), locs);
        pipeOp1.getBody().getBlocks().front().dump();
        llvm::outs() << "--------------------------------------------------------------------------------" << "\n";
        llvm::outs() << "--------------------------------------------------------------------------------" << "\n";

        //To latest pipeOp
        auto loc = pipeOp1->getLoc();
        auto pipelineOp = builder.create<mlir::daphne::VectorizedPipelineOp>(loc,
            mlir::ValueRange(results).getTypes(),
            operands,
            outRows,
            outCols,
            builder.getArrayAttr(vSplitAttrs),
            builder.getArrayAttr(vCombineAttrs),
            nullptr);
        mlir::Block *bodyBlock = builder.createBlock(&pipelineOp.getBody());

        //pipeOp1.getBody().getBlocks().splice(pipeOp1.getBody().end(), pip);

        /*auto pipelineOp = builder.create<mlir::daphne::VectorizedPipelineOp>(loc,
            mlir::ValueRange(results).getTypes(),
            operands,
            outRows,
            outCols,
            builder.getArrayAttr(vSplitAttrs),
            builder.getArrayAttr(vCombineAttrs),
            nullptr);
        mlir::Block *bodyBlock = builder.createBlock(&pipelineOp.getBody());*/
        


    }

    return;
}


std::unique_ptr<Pass> daphne::createHorizontalFusionPass() {
    return std::make_unique<HorizontalFusionPass>();
}