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
        ops.emplace_back(op);
    });
    std::reverse(ops.begin(), ops.end()); 

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
            decisionIxs.insert({op, 0});
            stack.push({op, nullptr, DisconnectReason::INVALID});
        }
    }

    VectorUtils::DEBUG::drawGraph(leafOps, "graph-gr2-pre.dot");

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

        auto vectOp = llvm::dyn_cast<daphne::Vectorizable>(op);

        for (size_t i = 0; i < vectOp->getNumOperands(); ++i) {
            auto operand = vectOp->getOperand(i);

            if (!llvm::isa<daphne::MatrixType>(operand.getType()))
                continue;

            //llvm::outs() << vectOp->getName().getStringRef().str() << " ";

            if (llvm::isa<mlir::BlockArgument>(operand)) {
                continue;
            }

            if (auto vectDefOp = llvm::dyn_cast<daphne::Vectorizable>(operand.getDefiningOp())) {
                auto numberOfDecisions = vectDefOp.getVectorCombines().size();

                //llvm::outs() << vectDefOp->getName().getStringRef().str() << "\n";

                bool foundMatch = false;
                for (VectorIndex vectDefOpIx = 0; vectDefOpIx < numberOfDecisions; ++vectDefOpIx) {

                    auto split = vectOp.getVectorSplits()[vectIx][i];
                    auto combine  = vectDefOp.getVectorCombines()[vectDefOpIx][0];

                    //same block missing
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
            } else {
                //defOp is outside of consideration, top horz. fusion possible
                //boundingOperations.push_back(op);
                //llvm::outs() << " test123\n";
            }
        }
    }

    VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, "graph-gr2-post.dot");

    //mmPCR to PCR
    std::map<PipelinePair, DisconnectReason> producerConsumerRelationships = VectorUtils::consolidateProducerConsumerRelationship(mmProducerConsumerRelationships); 

    //Topologoically greedy merge along the (valid) MULTIPLE_CONSUMER relationships
    bool change = true;
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
    }

    VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, "graph-gr2-pre-horz.dot");

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

    //identify horizintal fusions / scan sharing candidates

    //we need to know the operations of a pipeline that have somekind of input from outside (or another pipeline)
    std::map<Pipeline*, std::vector<mlir::Operation*>> boundingOpsOfPipelines;
    //TODO: this step could be combined with the intital dfs for identifiying pipelines
    //TODO: block arguments, what to do in this case, how do we know there are the same block arguments?
    for (auto pipe : pipelines) {
        std::vector<mlir::Operation*> boundingOpsOfPipe;
        for (auto op : *pipe) {
            for (auto operand : op->getOperands()) {
                if (auto defOp = operand.getDefiningOp()) {
                    if (operationToPipeline.find(defOp) != operationToPipeline.end()) {
                        if (operationToPipeline.at(defOp) != pipe) {
                            boundingOpsOfPipe.push_back(defOp); 
                        }
                    } 
                    else {
                        boundingOpsOfPipe.push_back(defOp); 
                    }
                }
            }
        }
        boundingOpsOfPipelines.insert({pipe, boundingOpsOfPipe});
    }

    //check for overlapping/intersection between the bounding ops
    std::vector<PipelinePair> horizontalRelationships;
    for (auto it1 = boundingOpsOfPipelines.begin(); it1 != boundingOpsOfPipelines.end(); ++it1) {
        auto& opsSet1 = it1->second;
        std::unordered_set<mlir::Operation*> opsUnique(opsSet1.begin(), opsSet1.end());

        for (auto it2 = next(it1); it2 != boundingOpsOfPipelines.end(); ++it2) {
            auto& opsSet2 = it2->second;

            // Check if there is any overlapping operation in the two sets.
            for (const auto& op : opsSet2) {
                if (opsUnique.find(op) != opsUnique.end()) {
                    horizontalRelationships.push_back({it1->first, it2->first});
                    break; // Break once found an overlapping operation in the current pair.
                }
            }
        }
    }

    /*//derive from existing producer consumer relationships
    std::vector<PipelinePair> horizontalRelationships;
    for(auto pipePairIt1 = producerConsumerRelationships.begin(); pipePairIt1 != producerConsumerRelationships.end(); ++pipePairIt1) {
        for(auto pipePairIt2 = std::next(pipePairIt1); pipePairIt2 != producerConsumerRelationships.end(); ++pipePairIt2) {
            //Must share the same pipeline as a producer (sufficient condition)
            //However this does not mean, that the individual ops also share inputs, where horz. Fusion makes sense (necessary condition)
            if(pipePairIt1->first.second == pipePairIt2->first.second) {
                horizontalRelationships.push_back({pipePairIt1->first.first, pipePairIt2->first.first});
            }
        }
    }*/
    //derive from top pipelines

    for(auto pipePair : horizontalRelationships) {
        auto pipe1 = pipePair.first;
        auto pipe2 = pipePair.second;

        //check if pipelines are connected somehow transitively
        //cannot merge a operation into another pipeline if it is connected somehow, as we basically decided against it earlier
        //TODO: combine
        if (VectorUtils::arePipelinesConnected(producerConsumerRelationships, pipe1, pipe2))
            continue;
        if (VectorUtils::arePipelinesConnected(producerConsumerRelationships, pipe2, pipe1))
            continue;

        VectorUtils::mergePipelines(pipelines, operationToPipeline, pipePair.first, pipePair.second);
    }

    VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, "graph-gr2.dot");

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