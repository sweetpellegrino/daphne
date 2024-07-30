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
        fillOp = nullptr;
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

    struct ThGreedyVectorizeComputationsPass : public PassWrapper<ThGreedyVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;

        //Function that modifies existing mlir programm structure
        //Currently only gets a pipeline as a list of nodes that | cf. Formalisation 
        void createVectorizedPipelineOps(func::FuncOp func, std::vector<std::vector<mlir::Operation *>> pipelines) {
            OpBuilder builder(func);

            /*for(auto pipeline : pipelines) {
            }*/
            
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

                    auto vSplits = std::vector<daphne::VectorSplit>();
                    auto vCombines = std::vector<daphne::VectorCombine>();
                    auto opsOutputSizes = std::vector<std::pair<Value, Value>>();
                    if (auto fillOp = llvm::dyn_cast<daphne::FillOp>(v)) {
                        vSplits = {daphne::VectorSplit::NONE, daphne::VectorSplit::NONE, daphne::VectorSplit::NONE};
                        vCombines = {daphne::VectorCombine::ROWS};

                        //Hard coded: instead use something like tryNumColsFromIthScalar?
                        //tryParamTraitUntil<u, tryNumColsFromIthScalar>::apply(numRows, numCols, op);
                        opsOutputSizes = {{fillOp.getOperands()[1], fillOp.getOperands()[2]}};

                        auto sizeTy = builder.getIndexType();
                        //create generator operation
                        /*auto loc = builder.getUnknownLoc();
                        auto gen = builder.create<daphne::GeneratorOp>(loc,
                            fillOp->getOperands()[0],
                            ArrayRef<Value>{fillOp.getOperands()[1]},
                            ArrayRef<Value>{fillOp.getOperands()[2]}
                        );
                        
                        auto numColOp = builder.create<daphne::NumColsOp>(
                            fillOp.getLoc(),
                            sizeTy,
                            ArrayRef<Value>{gen}
                        );
                        
                        auto numRowOp = builder.create<daphne::NumRowsOp>(
                            fillOp.getLoc(),
                            sizeTy,
                            ArrayRef<Value>{gen}
                        );

                        numColOp.dump();
                        numRowOp.dump();*/

                    } /*else if (v->hasTrait<mlir::OpTrait::VectorElementWise>()) {
                        vSplits.reserve(v->getNumOperands());
                        vCombines.reserve(v->getNumResults());
                        opsOutputSizes.reserve(v->getNumResults());
                            auto loc = v->getLoc();
                            auto sizeTy = builder.getIndexType();
                        for (auto operand : v->getOperands()) {
                            vSplits.push_back(operand.getType().template isa<daphne::MatrixType>() ? daphne::VectorSplit::ROWS : daphne::VectorSplit::NONE);
                        }

                        for (auto result : v->getResults()) {
                            vCombines.push_back(daphne::VectorCombine::ROWS);
                            opsOutputSizes.push_back({builder.create<daphne::NumRowsOp>(loc, sizeTy, v->getOperands()[0]),
                                                       builder.create<daphne::NumColsOp>(loc, sizeTy, v->getOperands()[0])});
                        }
                    }*/
                    else if (auto vec = llvm::dyn_cast<daphne::Vectorizable>(v)) {
                        vSplits = vec.getVectorSplits()[0];
                        vCombines = vec.getVectorCombines()[0];
                        opsOutputSizes = vec.createOpsOutputSizes(builder);
                    }

                    llvm::outs() << opsOutputSizes.size() << "\n";
                    for (auto vals : opsOutputSizes) {
                        llvm::outs() << "opOutputSize: ";
                        vals.first.dump();
                        vals.second.dump();
                        llvm::outs() << "\n";
                    }

                    // TODO: although we do create enum attributes, it might make sense/make it easier to
                    // just directly use an I64ArrayAttribute
                    // Determination of operands of VectorizedPipelineOps!
                    for(auto i = 0u; i < v->getNumOperands(); ++i) {
                        auto operand = v->getOperand(i);
                        if(!valueIsPartOfPipeline(operand)) {
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
                v->dump();
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
                            llvm::outs() << "Replacing " << nrowOp->getName() << "\n";
                            llvm::outs() << pipelineOp.getOutRows()[replacement.getResultNumber()] << "\n";
                            nrowOp.erase();
                        }
                        if(auto ncolOp = llvm::dyn_cast<daphne::NumColsOp>(op)) {
                            ncolOp.replaceAllUsesWith(pipelineOp.getOutCols()[replacement.getResultNumber()]);
                            llvm::outs() << "Replacing " << ncolOp->getName() << "\n";
                            llvm::outs() << pipelineOp.getOutCols()[replacement.getResultNumber()] << "\n";
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

void ThGreedyVectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    llvm::outs() << "TH_GREEDY" << "\n";

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
    std::vector<std::pair<mlir::Operation *, mlir::Operation *>> candidates;

    //reverse vectOps
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

        std::vector<decltype(candidates)::value_type> rel_candidates;
        std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(rel_candidates), [opv](const auto& c) {
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

    //print pipelines
    for (auto pipeline : pipelines) {
        llvm::outs() << "Pipeline: ";
        for (auto op : pipeline) {
            llvm::outs() << op->getName().getStringRef() << ",";
        }
        llvm::outs() << "\n";
    }

    //Step X: create pipeline ops
    ThGreedyVectorizeComputationsPass::createVectorizedPipelineOps(func, pipelines);

}

std::unique_ptr<Pass> daphne::createThGreedyVectorizeComputationsPass() {
    return std::make_unique<ThGreedyVectorizeComputationsPass>();
}