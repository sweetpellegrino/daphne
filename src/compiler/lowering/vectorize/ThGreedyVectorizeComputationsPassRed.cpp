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
#include "mlir/IR/Builders.h"
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

#include "compiler/lowering/vectorize/VectorizeComputationsUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace
{

    bool isVectorizable(Operation *op) {

        auto fillOp = llvm::dyn_cast<daphne::FillOp>(op);
        if ((op->hasTrait<mlir::OpTrait::VectorElementWise>() ||
            //op->hasTrait<mlir::OpTrait::VectorReduction>() ||
            op->hasTrait<mlir::OpTrait::VectorTranspose>() ||
            op->hasTrait<mlir::OpTrait::VectorMatMul>() ||
            llvm::dyn_cast<daphne::Vectorizable>(op)) &&
            fillOp == nullptr) {
                return true;
        }
        return false;
    }

    struct ThGreedyVectorizeComputationsPassRed : public PassWrapper<ThGreedyVectorizeComputationsPassRed, OperationPass<func::FuncOp>> {
        void runOnOperation() final;

        //Function that modifies existing mlir programm structure
        //Currently only gets a pipeline as a list of nodes that | cf. Formalisation
        void createVectorizedPipelineOps(func::FuncOp func, std::vector<std::vector<mlir::Operation *>> pipelines) {
            OpBuilder builder(func);

            // Create the `VectorizedPipelineOp`s
            for(auto pipeline : pipelines) {
                if(pipeline.empty()) {
                    continue;
                }
                auto valueIsPartOfPipeline = [&](Value operand) {
                    return llvm::any_of(pipeline, [&](mlir::Operation* lv) { return lv == operand.getDefiningOp(); });
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
                //looks at one operation at a time!
                for(auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
                    auto v = *vIt;
                    v->dump();

                    auto vSplits = std::vector<daphne::VectorSplit>();
                    auto vCombines = std::vector<daphne::VectorCombine>();
                    auto opsOutputSizes = std::vector<std::pair<Value, Value>>();
                    if (auto maxAgg = llvm::dyn_cast<daphne::AllAggSumOp>(v)) {
                        //probably need gen row and col
                        vSplits = {daphne::VectorSplit::ROWS};
                        vCombines = {daphne::VectorCombine::REDUCE};

                        auto loc = maxAgg->getLoc();
                        auto sizeTy = builder.getIndexType();
    
                        mlir::IntegerAttr valueAttr = builder.getIntegerAttr(sizeTy, 1);
                        mlir::Value value = builder.create<daphne::ConstantOp>(loc, sizeTy, valueAttr);

                        opsOutputSizes = {{value, value}};

                    }
                    else if (auto vec = llvm::dyn_cast<daphne::Vectorizable>(v)) {
                        vSplits = vec.getVectorSplits()[0];
                        vCombines = vec.getVectorCombines()[0];
                        opsOutputSizes = vec.createOpsOutputSizes(builder);
                    } 

                    // TODO: although we do create enum attributes, it might make sense/make it easier to
                    // just directly use an I64ArrayAttribute
                    // Determination of operands of VectorizedPipelineOps!
                    for(auto i = 0u; i < v->getNumOperands(); ++i) {
                        auto operand = v->getOperand(i);
                        operand.dump();
                        if(!valueIsPartOfPipeline(operand)){ //&& (!llvm::dyn_cast<daphne::NumRowsOp>(v) && !llvm::dyn_cast<daphne::NumColsOp>(v))) {
                            vSplitAttrs.push_back(daphne::VectorSplitAttr::get(&getContext(), vSplits[i]));
                            operands.push_back(operand);
                        }
                    }

                    // Determination of results of VectorizedPipelineOps!
                    for(auto vCombine : vCombines) {
                        vCombineAttrs.push_back(daphne::VectorCombineAttr::get(&getContext(), vCombine));
                    }
                    locations.push_back(v->getLoc());
                    for(auto result: v->getResults()) {
                        results.push_back(result);
                    }
                    for(auto outSize: opsOutputSizes) {
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

            //remove information from input matrices of pipeline
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
                    case daphne::VectorSplit::GEN: {
                        break;
                    }
                    case daphne::VectorSplit::NONE:
                        // keep any size information
                        break;
                }
                bodyBlock->addArgument(argTy, builder.getUnknownLoc());
            }

            llvm::outs() << "####5####\n";
            auto argsIx = 0u;
            auto resultsIx = 0u;
            //for every op in pipeline
            for(auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
                auto v = *vIt;
                auto numOperands = v->getNumOperands();
                auto numResults = v->getNumResults();

                //move v before end of block
                v->moveBefore(bodyBlock, bodyBlock->end());

                //set operands to arguments of body block, if defOp is not part of the pipeline
                for(auto i = 0u; i < numOperands; ++i) {
                    if(!valueIsPartOfPipeline(v->getOperand(i))) {
                        v->setOperand(i, bodyBlock->getArgument(argsIx++));
                    }
                }

                auto pipelineReplaceResults = pipelineOp->getResults().drop_front(resultsIx).take_front(numResults);
                resultsIx += numResults;
                for (auto z :
                    llvm::zip(v->getResults(), pipelineReplaceResults)) {
                auto old = std::get<0>(z);
                auto replacement = std::get<1>(z);

                // TODO: switch to type based size inference instead
                // FIXME: if output is dynamic sized, we can't do this
                // replace `NumRowOp` and `NumColOp`s for output size
                // inference
                for (auto &use : old.getUses()) {
                    auto *op = use.getOwner();
                    if (auto nrowOp = llvm::dyn_cast<daphne::NumRowsOp>(op)) {
                    auto test = llvm::dyn_cast<daphne::GeneratorOp>(
                        nrowOp.getArg().getDefiningOp());
                    if (!test) {
                        llvm::outs()
                            << "Replacing " << pipelineOp.getOutRows().size()
                            << " " << replacement.getResultNumber() << "\n";
                        nrowOp.replaceAllUsesWith(
                            pipelineOp
                                .getOutRows()[replacement.getResultNumber()]);
                        llvm::outs() << "Replacing ";
                        nrowOp->dump();
                        llvm::outs()
                            << "Replacement: "
                            << pipelineOp
                                .getOutRows()[replacement.getResultNumber()]
                            << "\n";
                        nrowOp.erase();
                    }
                    }
                    if (auto ncolOp = llvm::dyn_cast<daphne::NumColsOp>(op)) {
                    auto test = llvm::dyn_cast<daphne::GeneratorOp>(
                        ncolOp.getArg().getDefiningOp());
                    if (!test) {
                        llvm::outs()
                            << "Replacing " << pipelineOp.getOutRows().size()
                            << " " << replacement.getResultNumber() << "\n";
                        ncolOp.replaceAllUsesWith(
                            pipelineOp
                                .getOutCols()[replacement.getResultNumber()]);
                        llvm::outs() << "Replacing ";
                        ncolOp->dump();
                        llvm::outs()
                            << "Replacement: "
                            << pipelineOp
                                .getOutCols()[replacement.getResultNumber()]
                            << "\n";
                        ncolOp.erase();
                    }
                    }
                }
                // Replace only if not used by pipeline op
                old.replaceUsesWithIf(
                    replacement, [&](OpOperand &opOperand) {
                        llvm::outs() << "Op: ";
                        opOperand.getOwner()->dump();
                        bool test =
                            llvm::count(pipeline, opOperand.getOwner()) == 0;
                        llvm::outs() << "Test: " << test << "\n";
                        return test;
                    });
                }
                llvm::outs() << "###--###" << "\n";
                func.dump();
                llvm::outs() << "########" << "\n";
            }
            bodyBlock->walk([](Operation* op) {
                for(auto resVal: op->getResults()) {
                    if(auto ty = resVal.getType().dyn_cast<daphne::MatrixType>()) {
                        resVal.setType(ty.withShape(-1, -1));
                    }
                }
            });
            llvm::outs() << "####5####\n";
            builder.setInsertionPointToEnd(bodyBlock);
            //remove first two returns as they corresponds to the return of genshape numCol and numRows -> error
            //results.erase(results.begin(), results.begin() + 2);
            builder.create<daphne::ReturnOp>(loc, results);

            for ( auto resVal : results) {
                llvm::outs() << "Result: ";
                resVal.dump();
                resVal.getType().dump();
                //llvm::outs() << resVal.getType().getTypeID() <<  "\n";
                llvm::outs() << "\n";
            
            }
            }
        }
    };
}

void ThGreedyVectorizeComputationsPassRed::runOnOperation()
{

    auto func = getOperation();

    llvm::outs() << "TH_GREEDY_RED" << "\n";

    //Step 1: Filter vectorizbale operations
    //Slice Analysis for reverse topological sorting?
    //Good illustration here: PostOrder https://mlir.llvm.org/doxygen/SliceAnalysis_8h_source.html
    //Implementation of walk here: https://mlir.llvm.org/doxygen/Visitors_8h_source.html#l00062

    llvm::outs() << "######## STEP 1 ########" << "\n";

    std::vector<mlir::Operation *> vectOps;
    func->walk([&](mlir::Operation* op) {
        if(isVectorizable(op)) {
            vectOps.emplace_back(op);
        }
    });
    std::reverse(vectOps.begin(), vectOps.end());

    //print vectOps
    for (auto &op : vectOps) {
        llvm::outs() << "Op: ";
        op->dump();
    }

    llvm::outs() << "######## END ########" << "\n";

    //Improvment: can we already know that some operations can not be fused???
    //Step 2: Identify merging candidates
    llvm::outs() << "######## STEP 2 ########" << "\n";

    //TODO: what about duplicated candidates?
    using Candidate = std::pair<mlir::Operation *, mlir::Operation *>;
    std::vector<Candidate> candidates_producer_consumer;
    //TODO: Should this make a weak connection? So in case of not being greedy; first to broken up, if necessary
    //unordered set? with custom hash andqual function
    std::vector<Candidate> candidates_horizontal_consumers;

    //reversed vectOps
    for (auto &opv : vectOps) {

        //Get incoming edges of operation opv
        //One condition for Fusible: Check of producer <-> consumer relationship (opv->getOperand())
        //Improvement: For every operation of incoming argument of opv, check the additional conditions
        //True: push into possible merge candidates list
        for (size_t i = 0; i < opv->getNumOperands(); i++) {

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
            auto operand = opv->getOperand(i);
            //If not in a separate variable it does not work?
            //auto users = operand.getUsers();
            //for (auto user : users) {
            for (auto user : operand.getUsers()) {
                
                if (user == opv || //Does not make sense to consider the opv with itself
                    !isVectorizable(user) )//|| //User must be Vectorizable
                    //user->getBlock() == opv->getBlock()) //TODO: To restrictive?
                    continue;

                //We need to check if opv and user are not in a producer / consumer relationship
                bool is_only_horizontal = true;
                for (auto rel : user->getOperands()) {
                    if (rel.getDefiningOp() == opv) {
                        is_only_horizontal = false;
                        break;
                    }
                }

                if (is_only_horizontal) {
                    llvm::outs() << "H-Candidate: " << opv->getName() << " <-x-> " << user->getName() << "\n";
                    candidates_horizontal_consumers.push_back({opv, user});
                }
            }

            //-----------------------------------------------------------------
            // Producer -> Consumer
            //-----------------------------------------------------------------

            //Get producer of operand
            auto producer = operand.getDefiningOp();
            //Check if producer & consumer are in the same block
            if(isVectorizable(producer) && (producer->getBlock() == opv->getBlock())) {
                //Currently not needed: checking the split/combine.
                //cf. Algo
                llvm::outs() << "PC-Candidate: " << producer->getName() << " -> " << opv->getName() << "\n";
                candidates_producer_consumer.push_back({producer, opv});
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
    //Iteration over the individual vectOps allows for pipelines with size of one
    for(auto opv : vectOps) {
        //identify if opv is already part of a pipeline
        auto opv_it = operationToPipelineIx.find(opv);

        llvm::outs() << "######" << "\n";
        llvm::outs() << "opv: " << opv->getName().getStringRef() << "\n";
        if(opv_it == operationToPipelineIx.end()) {
            llvm::outs() << "opv_it == end" << "\n";
            std::vector<mlir::Operation*> pipeline;
            pipeline.push_back(opv);
            opv_it = operationToPipelineIx.insert({opv, pipelines.size()}).first;
            pipelines.push_back(pipeline);
        }

        size_t opv_pipeId = opv_it->second;
        llvm::outs() << "opv_pipeId: " << opv_pipeId << "\n";

        std::vector<decltype(candidates_producer_consumer)::value_type> rel_candidates;
        std::copy_if(candidates_producer_consumer.begin(), candidates_producer_consumer.end(), std::back_inserter(rel_candidates), [opv](const auto& c) {
            return (std::get<1>(c) == opv);
        });

        for (auto candidate : rel_candidates) {

            auto opi_it = operationToPipelineIx.find(std::get<0>(candidate));
            llvm::outs() << "opi: " << std::get<0>(candidate)->getName().getStringRef() << "\n";

            if (opi_it == operationToPipelineIx.end()) {
                pipelines.at(opv_pipeId).push_back(std::get<0>(candidate));
                operationToPipelineIx.insert({std::get<0>(candidate), opv_pipeId});
            }
            //merge both pipelines
            else {
                llvm::outs() << "merge both pipelines\n";
                size_t opi_pipeId = opi_it->second;
                std::vector<mlir::Operation*> mergedPipeline(pipelines.at(opv_pipeId));
                for (auto op : pipelines.at(opi_pipeId)) {
                    if  (std::find(mergedPipeline.begin(), mergedPipeline.end(), op) == mergedPipeline.end()) {
                        mergedPipeline.push_back(op);
                        operationToPipelineIx[op] = opv_pipeId;
                    }
                }
                pipelines.at(opv_pipeId) = std::move(mergedPipeline);
                pipelines.erase(pipelines.begin()+opi_pipeId);
            }
        }
        llvm::outs() << "######" << "\n";
    }

    llvm::outs() << "######## END ########" << "\n";

    //Step 4: Horizontal Fusion, if possible (utilize also at graph level)
    //Step 5: Small pipeline merge, if possible? why not further try to reduce the number of individual and merge together if constraints are met

    //print pipelines
    for (auto pipeline : pipelines) {
        llvm::outs() << "Pipeline: ";
        for (auto op : pipeline) {
            llvm::outs() << op->getName().getStringRef() << ",";
        }
        llvm::outs() << "\n";
    }

    //Step X: create pipeline ops
    ThGreedyVectorizeComputationsPassRed::createVectorizedPipelineOps(func, pipelines);

}

std::unique_ptr<Pass> daphne::createThGreedyVectorizeComputationsPassRed() {
    return std::make_unique<ThGreedyVectorizeComputationsPassRed>();
}
