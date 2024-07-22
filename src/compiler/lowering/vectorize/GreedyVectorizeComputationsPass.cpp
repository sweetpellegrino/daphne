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
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace
{
    /*
    bool isVectorizable(func::FuncOp &op) {
        return true;
    }

    bool isFusible(daphne::Vectorizable opi, daphne::Vectorizable opv) {
        if (opi->getBlock() != opv->getBlock())
            return false;

        return true;
    }
    */
    
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
    
    struct GreedyVectorizeComputationsPass : public PassWrapper<GreedyVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;


        //Function that modifies existing mlir programm structure
        //Currently only gets a pipeline as a list of nodes that | cf. Formalisation 
        void createVectorizedPipelineOps(func::FuncOp func, std::vector<std::vector<daphne::Vectorizable>> pipelines) {
            OpBuilder builder(func);
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

    //Split: before consumer
    //Combine: after producer
    //Split size: also relevant for tiles?
    struct VectorCandidateOption {
        daphne::VectorCombine combine;
        daphne::VectorSplit split;
        //double score; ??
        //size_t split_size; ???
        //size_t combine_size; ???
        //size_t tile_width;
        //size_t tile_height;
    };

    struct VectorCandidate {
        //std::shared_ptr<daphne::Vectorizable> producer;
        //std::shared_ptr<daphne::Vectorizable> consumer;
        daphne::Vectorizable producer;
        daphne::Vectorizable consumer;

        size_t argNumberOfConsumer = -1;

        //   a   f        a        a  f
        //  / \ /   =>    |  and   | /
        // b   c          b        c   
        //bool redundant = false; what about redudant computation?

        std::vector<VectorCandidateOption> options;
        std::vector<std::string> deps;
    };

    /*
    struct VectorPipelineCandidates {
        std::vector<daphne::Vectorizable> options; 
        std::vector<mlir::Result> isFusible; 
        std::vector<> isFusible; 
        std::vector<mlir::Result> isFusible; 
    }
    */

    struct VectorizationPlan {
        std::unordered_map<std::string, VectorCandidate> candidates;

        void addCandidate(const VectorCandidate& candidate) {
            auto key = getKey(candidate);
            if (candidates.find(key) == candidates.end()) {
                candidates[key] = candidate;
            } else {
                //merge?
                throw new std::runtime_error("Duplicate vectorization candidate");
            }
        }

        std::string getKey(const VectorCandidate& candidate) const {
            return candidate.producer->getName().getStringRef().str() + "_" + candidate.consumer->getName().getStringRef().str() + "_" + std::to_string(candidate.argNumberOfConsumer);
        }

        bool validateAndCollapse(const VectorCandidate& candidate) const {
            return false;
        }

        std::vector<std::vector<std::pair<daphne::Vectorizable, size_t>>> getPipelines() {
            return {};
        }

    };
    
    void printVC(const VectorCandidate& candidate) {
        llvm::outs() << "##################" << "\n";
        llvm::outs() << "Producer: ";
        candidate.producer->dump();
        llvm::outs() << "Consumer: ";
        candidate.consumer->dump();
        for (auto& option : candidate.options) {
            llvm::outs() << "Split: " << option.split << ", Combine: " << option.combine << "\n";
        }
        llvm::outs() << "##################" << "\n";
     }
}

void GreedyVectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    llvm::outs() << "GREEDY" << "\n";

    //Step 1: Reverse Topological Sorting & Filtering of Vectorizable functions
    std::vector<daphne::Vectorizable> vectOps;
    func->walk<WalkOrder::PostOrder>([&](mlir::Operation *op) {
        if(auto vec = llvm::dyn_cast<daphne::Vectorizable>(op)) {
            vectOps.push_back(vec);
            llvm::outs() << "vec func name: " << op->getName() << "\n";
        }
    });

    llvm::outs() << "######## Step 2 ########" << "\n";

    //Improvment: can we already know that some operations can not be fused??? 

    //Step 2: Identify merging candidates
    std::vector<VectorCandidate> candidates;
    for (auto &opv : vectOps) {

        llvm::outs() << "opv name: " << opv->getName() << "\n";

        //Get incoming edges of operation opv
        //Fusible: Check of producer<->consumer relationship (opv->getOperand())
        //For every operation of incoming argument of opv, check the additioanl conditions
        //True: push into possible merge candidates list
        for (size_t i = 0; i < opv->getNumOperands(); i++) {
            auto arg = opv->getOperand(i);
            llvm::outs() << "arg id: " << i << "\n";
            
            //Fusible: Check if both operations are vectorizable
            if(auto opi = arg.getDefiningOp<daphne::Vectorizable>()) {

                llvm::outs() << "opi name: " << opi->getName() << "\n";
                
                //Fusible: Check if both operations are in the same block
                if(opi->getBlock() != opv->getBlock())
                    continue;

                //flatten of split in VectorSplit list
                auto _splits = opv.getVectorSplits();
                std::vector<daphne::VectorSplit> splitsOfArgI;
                //from {{1,1},{2,2},{1,0}} to
                //e.g. for arg 0 it would be {1,2,1}
                //e.g. for arg 1 it would be {1,2,0}
                std::transform(_splits.begin(), _splits.end(), std::back_inserter(splitsOfArgI),
                    [i](const auto& vec) { return i < vec.size() ? vec[i] : daphne::VectorSplit::NONE; });

                //print splitsOfArgI
                llvm::outs() << "split: ";
                for (auto split : splitsOfArgI) {
                    llvm::outs() << (int) split << " ";
                }
                llvm::outs() << "\n";
                
                //Fusible: Determine the matching split, combines...
                //Vector splits should be about the arguments of opv, we need to map the argument to the position of the split in list
                std::vector<VectorCandidateOption> options;

                if(opi->hasTrait<mlir::OpTrait::VectorElementWise>()) {
                    if (opv->hasTrait<mlir::OpTrait::VectorElementWise>() || opv->hasTrait<mlir::OpTrait::VectorTranspose>() || opv->hasTrait<mlir::OpTrait::VectorReduction>()) {
                        options.push_back({daphne::VectorCombine::ROWS, daphne::VectorSplit::ROWS});
                        options.push_back({daphne::VectorCombine::COLS, daphne::VectorSplit::COLS});
                    }
                    else if (opv->hasTrait<mlir::OpTrait::VectorMatMul>()) {
                        options.push_back({});
                        options.push_back({});
                    }
                }
                else if (opv->hasTrait<mlir::OpTrait::VectorTranspose>()) {
                    if(opi->hasTrait<mlir::OpTrait::VectorElementWise>() || opi->hasTrait<mlir::OpTrait::VectorTranspose>()) {
                        options.push_back({daphne::VectorCombine::ROWS, daphne::VectorSplit::ROWS});
                        options.push_back({daphne::VectorCombine::COLS, daphne::VectorSplit::COLS});
                    }
                }
                
                /*
                for(auto split : splitsOfArgI) {

                    opv->dump();
                    opi.dump();

                    //flatten of combine in VectorCombine list
                    auto _combines = opi.getVectorCombines();
                    std::vector<daphne::VectorCombine> combineOfOPI;
                    std::transform(_combines.begin(), _combines.end(), std::back_inserter(combineOfOPI),
                        [i](const auto& vec) { return i < vec.size() ? vec[i] : throw std::exception(); });

                    for(auto combine : combineOfOPI) {
                        
                        llvm::outs() << ":" << (int) split << " " << (int) combine << "\n";

                        if ((int) split == (int) combine) {
                            VectorCandidateOption option {combine, split};
                            options.push_back(option);
                            break;
                        }
                    }
                }
                */
                if(!options.empty()) {
                    VectorCandidate candidate {opi, opv, i, options};
                    candidates.push_back(candidate);
                }
            }
        }
    }

    llvm::outs() << "######## END ########" << "\n";

    llvm::outs() << "VectorCandidates: " << candidates.size() << "\n";

    for(auto cand : candidates) {
        printVC(cand);
    }


    llvm::outs() << "######## Step 3 ########" << "\n";

    //Step 3: Greedy merge pipelines
    std::map<daphne::Vectorizable, size_t> operationToPipelineIx;
    std::vector<std::vector<daphne::Vectorizable>> pipelines;
    while(!candidates.empty()) {
        VectorCandidate candidate = candidates.back();
        candidates.pop_back();

        llvm::outs() << "Producer: " << candidate.producer->getName() << "\n";
        llvm::outs() << "Consumer: " << candidate.consumer->getName() << "\n";

        size_t consumerPipelineIndex = -1;
        for (size_t ix = 0; ix < pipelines.size(); ++ix) {
            auto& p = pipelines[ix];
            if(std::find(p.begin(), p.end(), candidate.consumer) != p.end()) {
                llvm::outs() << "Found consumer: " << candidate.consumer->getName() << "\n";
                consumerPipelineIndex = ix;
                break;
            }
        }

        if (consumerPipelineIndex == -1) {
            std::vector<daphne::Vectorizable> newPipeline = {candidate.consumer, candidate.producer};
            pipelines.push_back(newPipeline);
            operationToPipelineIx[candidate.producer] = pipelines.size() - 1;

        } else if (operationToPipelineIx.find(candidate.producer) == operationToPipelineIx.end()) { // Producer not found in any pipeline

            auto& consumerPipeline = pipelines[consumerPipelineIndex];
            consumerPipeline.push_back(candidate.producer);
            operationToPipelineIx[candidate.producer] = consumerPipelineIndex;
        }
    }

    for (auto p : pipelines) {
        llvm::outs() << "Pipeline: \n";
        for(auto v : p) {
            llvm::outs() << v->getName() << "\n";
        }
    }
    llvm::outs() << "######## END ########" << "\n";

    //Step X: create pipeline ops
    GreedyVectorizeComputationsPass::createVectorizedPipelineOps(func, pipelines);

}

std::unique_ptr<Pass> daphne::createGreedyVectorizeComputationsPass() {
    return std::make_unique<GreedyVectorizeComputationsPass>();
}