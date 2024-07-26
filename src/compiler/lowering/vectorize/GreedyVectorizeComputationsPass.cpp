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


#include "compiler/utils/CompilerUtils.h"
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <util/ErrorHandler.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <set>
#include <iostream>

#include <iostream>
#include <utility>
#include <vector>

#include "compiler/lowering/vectorize/VectorizeComputationsBase.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace
{

    bool isVectorizable(Operation *op) {

        if (op->hasTrait<mlir::OpTrait::VectorElementWise>() ||
            //op->hasTrait<mlir::OpTrait::VectorReduction>() ||
            op->hasTrait<mlir::OpTrait::VectorTranspose>() ||
            op->hasTrait<mlir::OpTrait::VectorMatMul>() ||
            llvm::dyn_cast<daphne::Vectorizable>(op)) {
                return true;
        }

        return false;
    }

    /**
     * @brief Recursive function checking if the given value is transitively dependant on the operation `op`.
     * @param value The value to check
     * @param op The operation to check
     * @return true if there is a dependency, false otherwise
     */
    bool valueDependsOnResultOf(Value value, Operation *op) {
        if (auto defOp = value.getDefiningOp()) {
            if (defOp == op)
                return true;
#if 1
            // TODO This crashes if defOp and op are not in the same block.
            // At the same time, it does not seem to be strictly required.
//            if (defOp->isBeforeInBlock(op))
            // Nevertheless, this modified line seems to be a good soft-filter;
            // without that, the vectorization pass may take very long on
            // programs with 100s of operations.
            if (defOp->getBlock() == op->getBlock() && defOp->isBeforeInBlock(op))
                // can't have results of `op` as inputs, as it is defined before
                return false;
#endif
            for (auto operand : defOp->getOperands()) {
                if (valueDependsOnResultOf(operand, op))
                    return true;
            }
        }
        return false;
    }

    /**
     * @brief Moves operation which are between the operations, which should be fused into a single pipeline, before
     * or after the position where the pipeline will be placed.
     * @param pipelinePosition The position where the pipeline will be
     * @param pipeline The pipeline for which this function should be executed
     */
    void movePipelineInterleavedOperations(Block::iterator pipelinePosition, const std::vector<daphne::Vectorizable> &pipeline) {
        // first operation in pipeline vector is last in IR, and the last is the first
        auto startPos = pipeline.back()->getIterator();
        auto endPos = pipeline.front()->getIterator();
        auto currSkip = pipeline.rbegin();
        std::vector<Operation*> moveBeforeOps;
        std::vector<Operation*> moveAfterOps;
        for(auto it = startPos; it != endPos; ++it) {
            if (it == (*currSkip)->getIterator()) {
                ++currSkip;
                continue;
            }

            bool dependsOnPipeline = false;
            auto pipelineOpsBeforeIt = currSkip;
            while (--pipelineOpsBeforeIt != pipeline.rbegin()) {
                for (auto operand : it->getOperands()) {
                    if(valueDependsOnResultOf(operand, *pipelineOpsBeforeIt)) {
                        dependsOnPipeline = true;
                        break;
                    }
                }
                if (dependsOnPipeline) {
                    break;
                }
            }
            // check first pipeline op
            for (auto operand : it->getOperands()) {
                if(valueDependsOnResultOf(operand, *pipelineOpsBeforeIt)) {
                    dependsOnPipeline = true;
                    break;
                }
            }
            if (dependsOnPipeline) {
                moveAfterOps.push_back(&(*it));
            }
            else {
                moveBeforeOps.push_back(&(*it));
            }
        }

        for(auto moveBeforeOp: moveBeforeOps) {
            moveBeforeOp->moveBefore(pipelinePosition->getBlock(), pipelinePosition);
        }
        for(auto moveAfterOp: moveAfterOps) {
            moveAfterOp->moveAfter(pipelinePosition->getBlock(), pipelinePosition);
            pipelinePosition = moveAfterOp->getIterator();
        }
    }

    bool pipelinesShareSameInputs(std::vector<mlir::Operation *> p1, std::vector<mlir::Operation *> p2) {
        //TODO: implement
    }
    
    struct GreedyVectorizeComputationsPass : public PassWrapper<GreedyVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;


        //Function that modifies existing mlir programm structure
        //Currently only gets a pipeline as a list of nodes that | cf. Formalisation 
        void createVectorizedPipelineOps(func::FuncOp func, std::vector<std::vector<mlir::Operation *>> op_pipelines) {
            OpBuilder builder(func);

            //make Opertion to daphne::Vectorizable
            std::vector<std::vector<daphne::Vectorizable>> pipelines;
            for(auto pipe : op_pipelines) {
                std::vector<daphne::Vectorizable> vecPipeline;
                for (auto it = pipe.rbegin(); it != pipe.rend(); ++it) {

                    auto op = *it;

                    llvm::outs() << "op: " << op->getName().getStringRef() << "\n";

                    auto vec = llvm::dyn_cast<daphne::Vectorizable>(op);
                    if (vec != nullptr) {
                        vecPipeline.push_back(vec);
                    }
                    else {
                        //TODO: ewAbs is not vectorizable
                        throw ErrorHandler::compilerError(
                                op, "GreedyVectorizeComputationsPass",
                                "ERROR #1: interface error @nik"
                                );
                    }
                }
                pipelines.push_back(vecPipeline);
            }

            // Create the `VectorizedPipelineOp`s
            for(auto pipeline : pipelines) {
                if(pipeline.empty()) {
                continue;
                }
                auto valueIsPartOfPipeline = [&](Value operand) {
                    return llvm::any_of(pipeline, [&](daphne::Vectorizable lv) { return lv == operand.getDefiningOp(); });
                };
                std::vector<Attribute> vSplitAttrs;
                std::vector<Attribute> vCombineAttrs;
                std::vector<Location> locations;
                std::vector<Value> results;
                std::vector<Value> operands;
                std::vector<Value> outRows;
                std::vector<Value> outCols;

                // first op in pipeline is last in IR
                builder.setInsertionPoint(pipeline.front());
                // move all operations, between the operations that will be part of the pipeline, before or after the
                // completed pipeline
                movePipelineInterleavedOperations(builder.getInsertionPoint(), pipeline);
                for(auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
                    auto v = *vIt;
                    auto vSplits = v.getVectorSplits()[0];
                    auto vCombines = v.getVectorCombines()[0];
                    // TODO: although we do create enum attributes, it might make sense/make it easier to
                    //  just directly use an I64ArrayAttribute
                    for(auto i = 0u; i < v->getNumOperands(); ++i) {
                        auto operand = v->getOperand(i);
                        if(!valueIsPartOfPipeline(operand)) {
                            vSplitAttrs.push_back(daphne::VectorSplitAttr::get(&getContext(), vSplits[i]));
                            operands.push_back(operand);
                        }
                    }
                    for(auto vCombine : vCombines) {
                        vCombineAttrs.push_back(daphne::VectorCombineAttr::get(&getContext(), vCombine));
                    }
                    locations.push_back(v->getLoc());
                    for(auto result: v->getResults()) {
                        results.push_back(result);
                    }
                    for(auto outSize: v.createOpsOutputSizes(builder)) {
                        llvm::outs() << "outsize: " << outSize.first << ", " << outSize.second << "\n";
                        outRows.push_back(outSize.first);
                        outCols.push_back(outSize.second);
                    }
                }
                
                std::vector<Location> locs;
                locs.reserve(pipeline.size());
                for(auto op: pipeline) {
                    locs.push_back(op->getLoc());
            }
            auto loc = builder.getFusedLoc(locs);
            auto pipelineOp = builder.create<daphne::VectorizedPipelineOp>(loc,
                ValueRange(results).getTypes(),
                operands,
                outRows,
                outCols,
                builder.getArrayAttr(vSplitAttrs),
                builder.getArrayAttr(vCombineAttrs),
                nullptr);
            Block *bodyBlock = builder.createBlock(&pipelineOp.getBody());

            for(size_t i = 0u; i < operands.size(); ++i) {
                auto argTy = operands[i].getType();
                switch (vSplitAttrs[i].cast<daphne::VectorSplitAttr>().getValue()) {
                    case daphne::VectorSplit::ROWS: {
                        auto matTy = argTy.cast<daphne::MatrixType>();
                        // only remove row information
                        argTy = matTy.withShape(-1, matTy.getNumCols());
                        break;
                    }
                    case daphne::VectorSplit::COLS: {
                        throw std::runtime_error("Not implemented");

                        /*auto matTy = argTy.cast<daphne::MatrixType>();
                        // only remove col information
                        argTy = matTy.withShape(matTy.getNumRows(), -1);
                        break;*/
                    }
                    case daphne::VectorSplit::NONE:
                        // keep any size information
                        break;
                }
                bodyBlock->addArgument(argTy, builder.getUnknownLoc());
            }

            auto argsIx = 0u;
            auto resultsIx = 0u;
            for(auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
                auto v = *vIt;
                auto numOperands = v->getNumOperands();
                auto numResults = v->getNumResults();

                v->moveBefore(bodyBlock, bodyBlock->end());

                for(auto i = 0u; i < numOperands; ++i) {
                    if(!valueIsPartOfPipeline(v->getOperand(i))) {
                        v->setOperand(i, bodyBlock->getArgument(argsIx++));
                    }
                }

                auto pipelineReplaceResults = pipelineOp->getResults().drop_front(resultsIx).take_front(numResults);
                resultsIx += numResults;
                for(auto z: llvm::zip(v->getResults(), pipelineReplaceResults)) {
                    auto old = std::get<0>(z);
                    auto replacement = std::get<1>(z);

                    // TODO: switch to type based size inference instead
                    // FIXME: if output is dynamic sized, we can't do this
                    // replace `NumRowOp` and `NumColOp`s for output size inference
                    for(auto& use: old.getUses()) {
                        auto* op = use.getOwner();
                        if(auto nrowOp = llvm::dyn_cast<daphne::NumRowsOp>(op)) {
                            nrowOp.replaceAllUsesWith(pipelineOp.getOutRows()[replacement.getResultNumber()]);
                            nrowOp.erase();
                        }
                        if(auto ncolOp = llvm::dyn_cast<daphne::NumColsOp>(op)) {
                            ncolOp.replaceAllUsesWith(pipelineOp.getOutCols()[replacement.getResultNumber()]);
                            ncolOp.erase();
                        }
                    }
                    // Replace only if not used by pipeline op
                    old.replaceUsesWithIf(replacement, [&](OpOperand& opOperand) {
                        return llvm::count(pipeline, opOperand.getOwner()) == 0;
                    });
                }
            }
            bodyBlock->walk([](Operation* op) {
                for(auto resVal: op->getResults()) {
                    if(auto ty = resVal.getType().dyn_cast<daphne::MatrixType>()) {
                        resVal.setType(ty.withShape(-1, -1));
                    }
                }
            });
            builder.setInsertionPointToEnd(bodyBlock);
            builder.create<daphne::ReturnOp>(loc, results);
            }
        }
    };
}

void GreedyVectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    llvm::outs() << "GREEDY" << "\n";

    //Step 1: Filter vectorizbale operations
    //Slice Analysis for reverse topological sorting?
    //Good illustration here: PostOrder https://mlir.llvm.org/doxygen/SliceAnalysis_8h_source.html
    //Implementation of walk here: https://mlir.llvm.org/doxygen/Visitors_8h_source.html#l00062

    llvm::outs() << "######## STEP 1 ########" << "\n";

    std::vector<mlir::Operation *> vectOps;
    func->walk([&](mlir::Operation* op)
    {
      if(isVectorizable(op))
          vectOps.emplace_back(op);
    });

    for (auto &opv : vectOps) {
        llvm::outs() << "Op: ";
        opv->dump();
        llvm::outs() << "\n";
    }

    llvm::outs() << "######## END ########" << "\n";

    //Improvment: can we already know that some operations can not be fused??? 
    //Step 2: Identify merging candidates
    llvm::outs() << "######## STEP 2 ########" << "\n";
    std::vector<std::pair<mlir::Operation *, mlir::Operation *>> candidates;

    //reverse vectOps
    std::reverse(vectOps.begin(), vectOps.end());
    for (auto &opv : vectOps) {
        
        //Get incoming edges of operation opv
        //One condition for Fusible: Check of producer<->consumer relationship (opv->getOperand())
        //Improvement: For every operation of incoming argument of opv, check the additional conditions
        //True: push into possible merge candidates list
        for (size_t i = 0; i < opv->getNumOperands(); i++) {

            //Get producer of operand
            auto producer = opv->getOperand(i).getDefiningOp();
            if(isVectorizable(producer)) {

                //Check if producer & consumer are in the same block
                if(producer->getBlock()!= opv->getBlock())
                    continue;
                //Currently not needed: checking the split/combine.
                //cf. Algo

                llvm::outs() << "Candidate: " << producer->getName() << " -> " << opv->getName() << "\n";
                candidates.push_back({producer, opv});
            }
        }
    }
    llvm::outs() << "######## END ########" << "\n";

    //Step 3: Greedy merge pipelines
    llvm::outs() << "######## STEP 3 ########" << "\n";
    
    // TODO: fuse pipelines that have the matching inputs, even if no output of the one pipeline is used by the other.
    // This requires multi-returns in way more cases, which is not implemented yet.
    std::map<mlir::Operation*, size_t> operationToPipelineIx;
    std::vector<std::vector<mlir::Operation*>> pipelines;
    while(!candidates.empty()) {
        auto candidate = candidates.back();
        candidates.pop_back();

        auto itProducer = operationToPipelineIx.find(candidate.first);
        auto itConsumer = operationToPipelineIx.find(candidate.second);

        // Line 22: Algo
        //Consumer and producer are not in a pipeline yet
        if (itProducer == operationToPipelineIx.end() && itConsumer == operationToPipelineIx.end()) {
            llvm::outs() << "both not in a pipeline" << "\n";
            pipelines.push_back({candidate.first, candidate.second});
            operationToPipelineIx[candidate.first] = pipelines.size() - 1;
            operationToPipelineIx[candidate.second] = pipelines.size() - 1;
        }
        // Line 28: Algo
        // {
        //Producer is in a pipeline, consumer not
        else if (itProducer != operationToPipelineIx.end() && itConsumer == operationToPipelineIx.end()) {
            llvm::outs() << "producer in a pipeline" << "\n";
            size_t ix = itProducer->second;
            //pipelines[ix].insert(pipelines[ix].begin(), candidate.second);
            pipelines[ix].push_back(candidate.second);
            operationToPipelineIx[candidate.second] = ix;
        }
        //Consumer is in a pipeline, producer not
        else if (itProducer == operationToPipelineIx.end() && itConsumer != operationToPipelineIx.end()) {
            llvm::outs() << "consumer in a pipeline" << "\n";
            size_t ix = itConsumer->second;
            pipelines[ix].push_back(candidate.first);
            operationToPipelineIx[candidate.first] = ix;
        }
        //Both are in a pipeline
        else if (itProducer != operationToPipelineIx.end() && itConsumer != operationToPipelineIx.end()) {
            size_t ixProducer = itProducer->second;
            size_t ixCosumer = itConsumer->second;
            llvm::outs() << ixProducer << " " << ixCosumer << "\n";
            llvm::outs() << "both in a pipeline" << "\n";
            if (ixProducer < ixCosumer) {
                pipelines[ixProducer].insert(pipelines[ixProducer].end(), pipelines[ixCosumer].begin(), pipelines[ixCosumer].end());
                for (auto& op : pipelines[ixCosumer]) {
                    operationToPipelineIx[op] = ixProducer;
                }
                pipelines.erase(pipelines.begin() + ixCosumer);
            }
            else if (ixCosumer < ixProducer) {
                pipelines[ixCosumer].insert(pipelines[ixCosumer].end(), pipelines[ixProducer].begin(), pipelines[ixProducer].end());
                for (auto& op : pipelines[ixProducer]) {
                    operationToPipelineIx[op] = ixCosumer;
                }
                pipelines.erase(pipelines.begin() + ixProducer);
            }
        }
        llvm::outs() << candidate.first->getName().getStringRef() << " -> " << candidate.second->getName().getStringRef() << "\n";
        // }
    }
    llvm::outs() << "######## END ########" << "\n";

    //remove all pipelines that only have one element
    //maxAll gets into a single pipeline how?
    //current mitigation
    auto it = pipelines.begin();
    while (it != pipelines.end()) {
        if (it->size() == 1) {
            it = pipelines.erase(it);
        } else {
            ++it;
        }
    }

    //print pipelines
    for (auto pipeline : pipelines) {
        llvm::outs() << "Pipeline: " << "\n";
        for (auto op : pipeline) {
            op->dump();
        }
    }

    //Step X: create pipeline ops
    GreedyVectorizeComputationsPass::createVectorizedPipelineOps(func, pipelines);

}

std::unique_ptr<Pass> daphne::createGreedyVectorizeComputationsPass() {
    return std::make_unique<GreedyVectorizeComputationsPass>();
}