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
    // CONST
    //-----------------------------------------------------------------

    const VectorIndex ZeroDecision = 0;

    //-----------------------------------------------------------------
    // Class functions
    //-----------------------------------------------------------------

    struct Greedy2VectorizeComputationsPass : public PassWrapper<Greedy2VectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
    }; 
}

    
void Greedy2VectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    std::vector<mlir::Operation*> ops;
    func->walk([&](daphne::Vectorizable op) {
        for (auto opType : op->getOperandTypes()) {
            if (!opType.isIntOrIndexOrFloat()) {
                ops.emplace_back(op);
                break;
            }
        }
    });
    std::reverse(ops.begin(), ops.end()); 

    /*for (auto op : ops) {
        VectorUtils::DEBUG::printVectorizableOperation(op);
    }*/

    //result
    std::vector<Pipeline*> pipelines;
    std::map<mlir::Operation*, size_t> decisionIxs;

    //helper
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
            decisionIxs.insert({op, 1});
            stack.push({op, nullptr, DisconnectReason::INVALID});
        }
    }

    VectorUtils::DEBUG::drawGraph(leafOps, "graph-gr2-null.dot");

    std::multimap<PipelinePair, DisconnectReason> mmProducerConsumerRelationships;
    std::map<mlir::Operation*, Pipeline*> operationToPipeline;

    while (!stack.empty()) {
        auto t = stack.top(); stack.pop();
        auto op = std::get<0>(t);
        auto currPipeline = std::get<1>(t);
        auto disReason = std::get<2>(t);

        if(operationToPipeline.find(op) != operationToPipeline.end()) {
            auto producerPipeline = operationToPipeline.at(op);
            mmProducerConsumerRelationships.insert({{currPipeline, producerPipeline}, disReason});
            continue;
        }

        if (disReason != DisconnectReason::NONE) {
            auto _pipeline = new Pipeline();
            pipelines.push_back(_pipeline);
            
            //check needed for empty init
            if (currPipeline != nullptr)
                mmProducerConsumerRelationships.insert({{currPipeline, _pipeline}, disReason});

            currPipeline = _pipeline;
        }

        operationToPipeline.insert({op, currPipeline});
        VectorIndex vectIx = decisionIxs.at(op);
        currPipeline->push_back(op);

        //llvm::outs() << "\n";
        auto vectOp = llvm::dyn_cast<daphne::Vectorizable>(op);

        for (int i = vectOp->getNumOperands() - 1; i >= 0; --i) {
        //for (size_t i = 0; i < vectOp->getNumOperands(); ++i) {
            auto operand = vectOp->getOperand(i);

            if (!llvm::isa<daphne::MatrixType>(operand.getType()))
                continue;

            //llvm::outs() << vectOp->getName().getStringRef().str() << " ";

            if (llvm::isa<mlir::BlockArgument>(operand)) {
                continue;
            }

            if (auto vectDefOp = llvm::dyn_cast<daphne::Vectorizable>(operand.getDefiningOp())) {
                auto numberOfDecisions = vectDefOp.getVectorCombines().size();

                auto split = vectOp.getVectorSplits()[vectIx][i];
                //llvm::outs() << vectDefOp->getName().getStringRef().str() << "\n";

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
    
                        //llvm::outs() << vectDefOpIx << ": " << "\n";

                        auto combine  = vectDefOp.getVectorCombines()[vectDefOpIx][0];

                        //llvm::outs() << split << " " << combine << "\n";

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
            else {
                //defOp is outside of consideration, top horz. fusion possible
                //boundingOperations.push_back(op);
                //llvm::outs() << "\n";
            }
        }
    }

    VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, "graph-gr2-step1.dot");

    //mmPCR to PCR
    std::map<PipelinePair, DisconnectReason> producerConsumerRelationships = VectorUtils::consolidateProducerConsumerRelationship(mmProducerConsumerRelationships); 
    //VectorUtils::DEBUG::printMMPCR(mmProducerConsumerRelationships);
    //VectorUtils::DEBUG::printPCR(producerConsumerRelationships);
    //VectorUtils::DEBUG::printPipelines(pipelines);


    //Topologoically greedy merge along the (valid) MULTIPLE_CONSUMER relationships
    //llvm::outs() << "-------------------------------------------------" << "\n";
    bool change = true;
    size_t count = 0;
    while (change) {
        change = false;
        
        std::multimap<PipelinePair, DisconnectReason> mmPCR;
        for (const auto& [pipePair, disReason] : producerConsumerRelationships) {

            if (disReason == DisconnectReason::INVALID)
                continue;

            if (VectorUtils::tryTopologicalSortMerged(pipelines, producerConsumerRelationships, pipePair.first, pipePair.second)) {
                auto mergedPipeline = VectorUtils::mergePipelines(pipelines, operationToPipeline, pipePair.first, pipePair.second);
                
                for (const auto& [_pipePair, _disReason] : producerConsumerRelationships) {

                    //Ignore in case that is current pair is pipePair 
                    if(_pipePair.first == pipePair.first && _pipePair.second == pipePair.second)
                        continue;

                    //Rewrite Relationships
                    if (_pipePair.first == pipePair.first || _pipePair.first == pipePair.second) {
                        auto newPipePair = std::make_pair(mergedPipeline, _pipePair.second);
                        mmPCR.insert({newPipePair, _disReason});
                    }
                    else if (_pipePair.second == pipePair.first || _pipePair.second == pipePair.second) {
                        auto newPipePair = std::make_pair(_pipePair.first, mergedPipeline);
                        mmPCR.insert({newPipePair, _disReason});
                    }
                    else { 
                        mmPCR.insert({_pipePair, _disReason});
                    }
                }

                change = true;
                break;
            }
        }

        //In case of no change the mmPCR is not filled, ignore
        if(change)
            producerConsumerRelationships = VectorUtils::consolidateProducerConsumerRelationship(mmPCR);

        std::ostringstream oss;
        oss << "graph-gr2-step2-" << count << ".dot";

        VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, oss.str());
        //VectorUtils::DEBUG::printPCR(producerConsumerRelationships);
        //VectorUtils::DEBUG::printPipelines(pipelines);

        count++;
    }

    VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, "graph-gr2-step2.dot");
    //VectorUtils::DEBUG::printPCR(producerConsumerRelationships);
    //VectorUtils::DEBUG::printPipelines(pipelines);

    //Post Processing

    std::vector<Pipeline> _pipelines;
    _pipelines.resize(pipelines.size());

    std::transform(pipelines.begin(), pipelines.end(), _pipelines.begin(), [](const auto& ptr) { return *ptr; }); 

    //will crash if for some reason the pipelines itself are not topologically sorted 
    VectorUtils::createVectorizedPipelineOps(func, _pipelines, decisionIxs);

    return;
}


std::unique_ptr<Pass> daphne::createGreedy2VectorizeComputationsPass() {
    return std::make_unique<Greedy2VectorizeComputationsPass>();
}