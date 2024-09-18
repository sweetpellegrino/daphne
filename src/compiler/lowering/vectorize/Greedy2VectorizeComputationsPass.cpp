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

    llvm::outs() << "Greedy2VectorizeComputationsPass" << "\n";

    std::vector<mlir::Operation*> ops;
    func->walk([&](daphne::Vectorizable op) {
        ops.emplace_back(op);
    });
    std::reverse(ops.begin(), ops.end());

    std::unordered_set<FusionPair> producerConsumerCandidates;
    std::unordered_set<FusionPair> horizontalCandidates;
    std::map<mlir::Operation*, VectorIndex> decisionIxs;

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
                if(auto vectProducer = llvm::dyn_cast<daphne::Vectorizable>(producer)) { 

                    auto split = vectOp.getVectorSplits()[ZeroDecision][i];

                    //currently only considering the first result of an operation
                    auto combine = vectProducer.getVectorCombines()[ZeroDecision][0];

                    if(VectorUtils::matchingVectorSplitCombine(split, combine) && 
                        (producer->getBlock() == vectOp->getBlock())) {
                    
                        //spdlog::debug("PC-Candidate: {} -> {}", producer->getName(), opv->getName());
                        //llvm::outs() << "PC-Candidate: " << producer->getName() << " " << vectOp->getName() << "\n";
                        producerConsumerCandidates.insert({producer, op, FusionPair::Type::PRODUCER_CONSUMER});
                    }
                }
            }
        }
    }

    std::map<mlir::Operation*, size_t> operationToPipelineIx;
    std::vector<std::vector<mlir::Operation*>> pipelines;

    //Iteration over the individual vectOps allows for pipelines with size of one
    for(const auto& op : ops) {

        //For this Greedy algorithm it is predetermined by ZeroDecision
        decisionIxs.insert({op, ZeroDecision});

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

        for (const auto& pcCand : relatedCandidates) {

            auto producerIt = operationToPipelineIx.find(pcCand.op1);

            if (producerIt == operationToPipelineIx.end()) {
                pipelines.at(opPipeIx).push_back(pcCand.op1);
                operationToPipelineIx.insert({pcCand.op1, opPipeIx});
            }
            else {
                size_t producerPipeIx = producerIt->second;
                if (opPipeIx != producerPipeIx) {
                    VectorUtils::mergePipelines(pipelines, operationToPipelineIx, opPipeIx, producerPipeIx);
                }
            }
        }
    }

    //Step 4: Horizontal Fusion
    //Separate step as it allows for the producer -> consumer relationship to be exploited first
    //What about size information and broadcast of the sharing operator: does it make sense if matrix too small? all inputs need to be
    for (const auto& hCand : horizontalCandidates) {
        
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
    }

    //debugging
    std::map<mlir::Operation*, VectorIndex> decisionIx;
    VectorUtils::DEBUG::printPipelines(ops, operationToPipelineIx, decisionIxs, "graph-gr2.dot");

    VectorUtils::createVectorizedPipelineOps(func, pipelines, decisionIxs);
}


std::unique_ptr<Pass> daphne::createGreedy2VectorizeComputationsPass() {
    return std::make_unique<Greedy2VectorizeComputationsPass>();
}