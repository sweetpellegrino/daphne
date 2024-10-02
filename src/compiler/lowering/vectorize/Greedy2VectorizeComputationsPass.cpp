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

    struct Greedy2VectorizeComputationsPass : public PassWrapper<Greedy2VectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
    }; 

    //-----------------------------------------------------------------
    // Helper
    //-----------------------------------------------------------------

    void greedyFindMinimumPipelines(std::stack<std::tuple<mlir::Operation*, Pipeline*, DisconnectReason>>& stack, std::vector<Pipeline*> &pipelines,  
        std::map<mlir::Operation*, Pipeline*> &operationToPipeline, std::multimap<PipelinePair, DisconnectReason> &mmProducerConsumerRelationships,
        std::map<mlir::Operation*, size_t> &decisionIxs) {

        while (!stack.empty()) {
            auto t = stack.top(); stack.pop();
            auto op = std::get<0>(t);
            auto currPipeline = std::get<1>(t);
            auto disReason = std::get<2>(t);

            // Operation was already visited.
            if(operationToPipeline.find(op) != operationToPipeline.end()) {
                auto producerPipeline = operationToPipeline.at(op);
                mmProducerConsumerRelationships.insert({{currPipeline, producerPipeline}, disReason});
                continue;
            }

            if (disReason != DisconnectReason::NONE) {
                auto _pipeline = new Pipeline();
                pipelines.push_back(_pipeline);
                
                // Check needed as initially the first element on stack does not have any precursor pipeline.
                if (currPipeline != nullptr)
                    mmProducerConsumerRelationships.insert({{currPipeline, _pipeline}, disReason});

                currPipeline = _pipeline;
            }

            operationToPipeline.insert({op, currPipeline});
            VectorIndex vectIx = decisionIxs.at(op);
            currPipeline->push_back(op);

            auto vectOp = llvm::dyn_cast<daphne::Vectorizable>(op);

            //for (int i = vectOp->getNumOperands() - 1; i >= 0; --i) {
            for (size_t i = 0; i < vectOp->getNumOperands(); ++i) {
                auto operand = vectOp->getOperand(i);

                if (!llvm::isa<daphne::MatrixType>(operand.getType()))
                    continue;

                if (llvm::isa<mlir::BlockArgument>(operand)) {
                    continue;
                }

                if (auto vectDefOp = llvm::dyn_cast<daphne::Vectorizable>(operand.getDefiningOp())) {
                    auto numberOfDecisions = vectDefOp.getVectorCombines().size();

                    auto split = vectOp.getVectorSplits()[vectIx][i];

                    if (decisionIxs.find(vectDefOp) != decisionIxs.end()) {
                        auto combine  = vectDefOp.getVectorCombines()[decisionIxs.at(vectDefOp)][0];
                        if (VectorUtils::matchingVectorSplitCombine(split, combine) && vectDefOp->getBlock() == vectOp->getBlock())
                            stack.push({vectDefOp, currPipeline, DisconnectReason::MULTIPLE_CONSUMERS});
                        else
                            stack.push({vectDefOp, currPipeline, DisconnectReason::INVALID});
                    } 
                    else {
                        bool foundMatch = false;
                        for (VectorIndex vectDefOpIx = 0; vectDefOpIx < numberOfDecisions; ++vectDefOpIx) {
        
                            auto combine  = vectDefOp.getVectorCombines()[vectDefOpIx][0];

                            if (VectorUtils::matchingVectorSplitCombine(split, combine) && vectDefOp->getBlock() == vectOp->getBlock()) {
                                if (vectDefOp->hasOneUse()) {
                                    stack.push({vectDefOp, currPipeline, DisconnectReason::NONE});
                                }
                                else {
                                    stack.push({vectDefOp, currPipeline, DisconnectReason::MULTIPLE_CONSUMERS});
                                }
                                decisionIxs.insert({vectDefOp, vectDefOpIx});
                                foundMatch = true;
                                break;
                            }
                        }
                        if(!foundMatch) {
                            decisionIxs.insert({vectDefOp, 0});
                            stack.push({vectDefOp, currPipeline, DisconnectReason::INVALID});
                        }
                    }
                } 
            }
        }
    }

}

    
void Greedy2VectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    std::vector<mlir::Operation*> ops;
    func->walk([&](daphne::Vectorizable op) {
        for (auto opType : op->getOperandTypes()) {
            if (!opType.isIntOrIndexOrFloat() && !llvm::isa<daphne::StringType>(opType)) {
                ops.emplace_back(op);
                break;
            }
        }
    });
    std::reverse(ops.begin(), ops.end()); 

    std::vector<VectorIndex> startingDecisions = {0, 1};

    std::vector<std::vector<Pipeline*>> solutions;
    solutions.reserve(startingDecisions.size());

    std::vector<std::map<mlir::Operation*, VectorIndex>> solutions_decisionIxs;
    solutions_decisionIxs.reserve(startingDecisions.size());

    for (size_t i = 0; i < startingDecisions.size(); ++i) {
        auto startVectIndex = startingDecisions[i];

        std::vector<Pipeline*> pipelines;
        std::map<mlir::Operation*, size_t> decisionIxs;

        std::vector<mlir::Operation*> leafOps;
        std::stack<std::tuple<mlir::Operation*, Pipeline*, DisconnectReason>> stack;

        for (const auto &op : ops) {
            auto users = op->getUsers();
            bool found = false;
            for (auto u : users) {
                if (std::find(ops.begin(), ops.end(), u) != ops.end()) { 
                    found = true;
                    break;
                }
            }
            if(!found) {
                leafOps.push_back(op);
                auto vectOp = llvm::cast<daphne::Vectorizable>(op);
                
                // Fallback to Zero
                if (vectOp.getVectorCombines().size() <= startVectIndex)
                    decisionIxs.insert({op, 0});
                else
                    decisionIxs.insert({op, startVectIndex});
                stack.push({op, nullptr, DisconnectReason::INVALID});
            }
        }

        std::multimap<PipelinePair, DisconnectReason> mmProducerConsumerRelationships;
        std::map<mlir::Operation*, Pipeline*> operationToPipeline;

        // Step 1
        greedyFindMinimumPipelines(stack, pipelines, operationToPipeline, mmProducerConsumerRelationships, decisionIxs);

        // Aggreagate
        std::map<PipelinePair, DisconnectReason> producerConsumerRelationships = VectorUtils::consolidateProducerConsumerRelationship(mmProducerConsumerRelationships); 

        // Step 2
        VectorUtils::greedyMergePipelinesProducerConsumer(pipelines, operationToPipeline, producerConsumerRelationships);

        solutions.push_back(pipelines);
        solutions_decisionIxs.push_back(decisionIxs);

    }

    //TODO improve
    size_t min = 0;
    size_t min_size = INFINITY;
    for (size_t i = 0; i < solutions.size(); ++i){
        size_t size = solutions.at(i).size();
        if(min_size > size) {
            min = i;
            min_size = size;
        }
    }

    std::vector<Pipeline*> pipelines = solutions.at(min);
    std::map<mlir::Operation*, VectorIndex> decisionIxs = solutions_decisionIxs.at(min);

    // Post Processing
    std::vector<Pipeline> _pipelines;
    _pipelines.resize(pipelines.size());
    std::transform(pipelines.begin(), pipelines.end(), _pipelines.begin(), [](const auto& ptr) { return *ptr; }); 

    VectorUtils::createVectorizedPipelineOps(func, _pipelines, decisionIxs);

    return;
}


std::unique_ptr<Pass> daphne::createGreedy2VectorizeComputationsPass() {
    return std::make_unique<Greedy2VectorizeComputationsPass>();
}