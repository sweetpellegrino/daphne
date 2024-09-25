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
#include <cstdint>
#include <mlir/IR/OpDefinition.h>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "spdlog/fmt/bundled/core.h"

#include <algorithm>
#include <sstream>
#include <stack>
#include <string>
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

    struct Greedy1VectorizeComputationsPass : public PassWrapper<Greedy1VectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
    }; 

    void printStack(std::stack<std::tuple<mlir::Operation*, Pipeline*, Pipeline*>> s) {
        llvm::outs() << "[";
        while (!s.empty()) {
            auto op = s.top();
            llvm::outs() << "(" << std::get<0>(op)->getName().getStringRef().str() << ", " << std::get<1>(op) << "), ";
            s.pop();
        }
        llvm::outs() << "]\n";
    }

    void printGraph(std::vector<mlir::Operation*> leafOps, std::string filename) {
        std::stack<mlir::Operation*> stack;
        std::ofstream dot(filename);
        if (!dot.is_open()) {
            throw std::runtime_error("test");
        }

        dot << "digraph G {\n";
        for (auto leaf : leafOps) {
            stack.push(leaf);
        }

        std::vector<mlir::Operation*> visited;

        while (!stack.empty()) {
            auto op = stack.top(); stack.pop();
            if(std::find(visited.begin(), visited.end(), op) != visited.end()) {
                continue;
            }
            visited.push_back(op);

            auto v = llvm::dyn_cast<daphne::Vectorizable>(op);
            for (unsigned i = 0; i < v->getNumOperands(); ++i) {
                mlir::Value e = v->getOperand(i);
                auto defOp = e.getDefiningOp();
                if (llvm::isa<daphne::MatrixType>(e.getType()) && llvm::isa<daphne::Vectorizable>(defOp)) {
                    dot << "\"" << defOp->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(defOp) << "\" -> \"" << op->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(op) << "\" [label=\"" << i << "\"];\n";
                    stack.push(defOp);
                }
            }
        }
        dot << "}";
        dot.close();
    }

 

}

void Greedy1VectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    std::vector<mlir::Operation*> ops;
    func->walk([&](daphne::Vectorizable op) {
        ops.emplace_back(op);
    });
    std::reverse(ops.begin(), ops.end()); 

    //result
    std::vector<Pipeline*> pipelines;
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
            stack.push({op, nullptr, DisconnectReason::INVALID});
        }
    }

    VectorUtils::DEBUG::drawGraph(leafOps, "graph-gr1-pre.dot");

    std::multimap<PipelinePair, DisconnectReason> mmProducerConsumerRelationships;
    std::map<mlir::Operation*, Pipeline*> operationToPipeline;

    //std::vector<mlir::Operation*> boundingOperations;

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
                //llvm::outs() << vectDefOp->getName().getStringRef().str() << "\n";

                auto split = vectOp.getVectorSplits()[ZeroDecision][i];
                auto combine  = vectDefOp.getVectorCombines()[ZeroDecision][0];

                //same block missing
                if (VectorUtils::matchingVectorSplitCombine(split, combine) && vectDefOp->getBlock() == vectOp->getBlock()) {
                    if (vectDefOp->hasOneUse()) {
                        stack.push({vectDefOp, currPipeline, DisconnectReason::NONE});
                    }
                    else {
                        stack.push({vectDefOp, currPipeline, DisconnectReason::MULTIPLE_CONSUMERS});
                    }
                }
                else {
                    stack.push({vectDefOp, currPipeline, DisconnectReason::INVALID});
                }
            } else {
                //defOp is outside of consideration, top horz. fusion possible
                //boundingOperations.push_back(op);
                //llvm::outs() << " test123\n";
            }
        }
    }

    //Needed as Greedy1 is only considering the first possiblity
    std::map<mlir::Operation*, size_t> decisionIxs;
    for (const auto& op : ops) {
        decisionIxs.insert({op, ZeroDecision});
    }
    
    VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, "graph-gr1-post.dot");

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

    VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, "graph-gr1-pre-horz.dot");

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

        if (pipePair.first->size() > pipePair.second->size()) {
            VectorUtils::mergePipelines(pipelines, operationToPipeline, pipePair.first, pipePair.second);
        }
        else {
            VectorUtils::mergePipelines(pipelines, operationToPipeline, pipePair.second, pipePair.first);
        }
    }

    VectorUtils::DEBUG::drawPipelines(ops, operationToPipeline, decisionIxs, "graph-gr1.dot");

    //Post Processing

    std::vector<Pipeline> _pipelines;
    _pipelines.resize(pipelines.size());

    std::transform(pipelines.begin(), pipelines.end(), _pipelines.begin(), [](const auto& ptr) { return *ptr; }); 

    //will crash if for some reason the pipelines itself are not topologically sorted 
    VectorUtils::createVectorizedPipelineOps(func, _pipelines, decisionIxs);

    return;
#if 0
    for (const auto &op : ops) {

        decisionIxs.insert({op, ZeroDecision});
        auto vectOp = llvm::dyn_cast<daphne::Vectorizable>(op);

        //Get incoming edges of operation opv
        //One condition for Fusible: Check of producer -> consumer relationship (opv->getOperand())
        //Improvement: For every operation of incoming argument of opv, check the additional conditions
        //True: push into possible merge candidates list

        for (size_t i = 0; i < vectOp->getNumOperands(); i++) {

            auto operand = vectOp->getOperand(i);
            
            //A scalar producer->consumer relationship is currently not supported (Reduction is nonetheless a pipeline breaker)
            //We do not want to fuse under a horizontal relationship in case we have scalar values (questionable improvement)
            if(llvm::isa<daphne::MatrixType>(operand.getType())) {
                //TODO: Do I need to check whether the operands are even from object type?
                //e.g. what would it mean, if the opv and user shares a constantOp result?

                //-----------------------------------------------------------------
                // Consumer <- Producer -> Consumer
                //-----------------------------------------------------------------

                //Based on the operand, check if the operand is also used from another operation
                //If yes, these are potentially candidates for horizontal fusion
                //Horizontal Fusion:
                //
                //          producer
                //         /        \
                //        opv       user
                //
                // => (opv, user)
                //TODO: What about the producer itself? if it is vectorizable, both vectorizable consumers will probably also land into same pipeline anyway?
                //Optimize by flipping order and early exist if producer, consumer relationship was created
                for (const auto &user : operand.getUsers()) {
                    
                    auto vectUser = llvm::dyn_cast<daphne::Vectorizable>(user);
                    if (!vectUser|| //Does not make sense to consider the opv with itself
                        vectUser == vectOp || //User must be Vectorizable
                        vectUser->getBlock() != vectOp->getBlock()) //TODO: To restrictive?
                        continue;

                    //We need to check if opv and user are not in a producer / consumer relationship
                    bool is_only_horizontal = true;
                    for (const auto &rel : user->getOperands()) {
                        if (rel.getDefiningOp() == vectOp) {
                            is_only_horizontal = false;
                            break;
                        }
                    }

                    size_t userOperandIx = 0;
                    for (const auto &use : user->getOperands()) {
                        if (use == operand) {
                            break;
                        }
                        userOperandIx++;
                    }


                    if (is_only_horizontal) {
                        if (vectOp.getVectorSplits()[ZeroDecision][i] == vectUser.getVectorSplits()[ZeroDecision][userOperandIx]) {
                            //spdlog::debug("H-Candidate: {} <-x-> {}", opv->getName(), user->getName());
                            //llvm::outs() << "H-Candidate: " << v_user->getName() << " " << opv->getName() << "\n";
                            horizontalCandidates.insert({op, user, FusionPair::Type::HORIZONTAL});
                        }
                    }
                }

                //-----------------------------------------------------------------
                // Producer -> Consumer
                //-----------------------------------------------------------------
                //Based on the operand, check if the operand is also used from another operation
                //If yes, these are potentially candidates for horizontal fusion
                //Horizontal Fusion:
                //
                //          producer
                //         /        \
                //        opv       user
                //
                // => (opv, user)

                //check if sufficient for identifiying the control flow  
                if (llvm::isa<mlir::BlockArgument>(operand))
                    continue;

                auto producer = operand.getDefiningOp();
                if((llvm::dyn_cast<daphne::Vectorizable>(producer)) && (producer->getBlock() == vectOp->getBlock())) { 

                    auto vectProducer = llvm::dyn_cast<daphne::Vectorizable>(producer);
                    auto split = vectOp.getVectorSplits()[ZeroDecision][i];

                    //currently only considering the first result of an operation
                    auto combine = vectProducer.getVectorCombines()[ZeroDecision][0];

                    if(VectorUtils::matchingVectorSplitCombine(split, combine)) {
                    
                        //spdlog::debug("PC-Candidate: {} -> {}", producer->getName(), opv->getName());
                        //llvm::outs() << "PC-Candidate: " << producer->getName() << " " << vectOp->getName() << "\n";
                        producerConsumerCandidates.insert({producer, op, FusionPair::Type::PRODUCER_CONSUMER});
                    }
                }
                else {
                    //Starting nodes with inputs from outside (decision is still needed)
                    producerConsumerCandidates.insert({nullptr, op,FusionPair::Type::PRODUCER_CONSUMER});
                }
            }
        }
    }

    for (auto pcPair : producerConsumerCandidates) {
        pcPair.print();
    }

    for (auto hPair : horizontalCandidates) {
        hPair.print();
    }

    std::map<mlir::Operation*, size_t> operationToPipelineIx;
    std::vector<std::vector<mlir::Operation*>> pipelines;

    //Iteration over the individual vectOps allows for pipelines with size of one

    for(const auto& op : ops) {

        //For this Greedy algorithm it is predetermined by ZeroDecision

        llvm::outs() << "\n";
        llvm::outs() << "Start" << "\n";
        op->print(llvm::outs());
        llvm::outs() << "\n";

        auto opIt = operationToPipelineIx.find(op);

        if(opIt == operationToPipelineIx.end()) {
            std::vector<mlir::Operation*> pipeline;
            pipeline.push_back(op);
            opIt = operationToPipelineIx.insert({op, pipelines.size()}).first;
            pipelines.push_back(pipeline);
        }

        size_t opPipeIx = opIt->second;

        std::vector<decltype(producerConsumerCandidates)::value_type> relatedCandidates;
        std::copy_if(producerConsumerCandidates.begin(), producerConsumerCandidates.end(), std::back_inserter(relatedCandidates), [op](const auto& c) {
            return (c.op2 == op);
        });

        llvm::outs() << "Rel" << "\n";
        for (auto r : relatedCandidates) {
            r.print();
        }
        llvm::outs() << "\n";


        for (const auto& pcCand : relatedCandidates) {
            
            if (pcCand.op1 == nullptr)
                continue;

            auto producerIt = operationToPipelineIx.find(pcCand.op1);
            //size_t opPipeIx = operationToPipelineIx.find(pcCand.op2)->second;

            if (producerIt == operationToPipelineIx.end()) {
                pipelines.at(opPipeIx).push_back(pcCand.op1);
                operationToPipelineIx.insert({pcCand.op1, opPipeIx});
            }
            else {
                size_t producerPipeIx = producerIt->second;
                llvm::outs() << opPipeIx << " " << producerPipeIx << "\n";
                if (VectorUtils::arePipelinesConnected(pipelines, operationToPipelineIx, opPipeIx, producerPipeIx)) {
                    if (VectorUtils::tryTopologicalSortPreMergedPipelines(pipelines, operationToPipelineIx, opPipeIx, producerPipeIx)) {
                        llvm::outs() << "merge" << "\n";
                        opPipeIx = VectorUtils::mergePipelines(pipelines, operationToPipelineIx, opPipeIx, producerPipeIx);
                    }
                }
            }

            for (size_t i = 0; i < pipelines.size(); ++i) {
                llvm::outs() << "Pipe " << i << ": ";
                for (auto _op : pipelines.at(i)) {
                    llvm::outs() << _op->getName().getStringRef().str() << ", ";
                }
                llvm::outs() << "\n";
            }

        }
    }

    //Step 4: Horizontal Fusion
    //Separate step as it allows for the producer -> consumer relationship to be exploited first
    //What about size information and broadcast of the sharing operator: does it make sense if matrix too small? all inputs need to be
    /*for (const auto& hCand : horizontalCandidates) {
        
        auto op1Ix = operationToPipelineIx.find(hCand.op1)->second;
        auto op2Ix = operationToPipelineIx.find(hCand.op2)->second;

        // Check if id is identical, if yes do nothing
        if (op1Ix == op2Ix)
            continue;

        //in case of possiblity check for interconnectivenes
        //cannot merge a operation into another pipeline if it is connected somehow => reason earlier we basically decided against
        if (VectorUtils::arePipelinesConnected(pipelines, operationToPipelineIx, op1Ix, op2Ix))
            continue;

        VectorUtils::mergePipelines(pipelines, operationToPipelineIx, op1Ix, op2Ix);
    }*/
#endif

}


std::unique_ptr<Pass> daphne::createGreedy1VectorizeComputationsPass() {
    return std::make_unique<Greedy1VectorizeComputationsPass>();
}