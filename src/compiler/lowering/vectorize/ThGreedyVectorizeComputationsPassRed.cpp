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

#include <cstddef>
#include <cstdint>
#include <map>
#include <functional>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <util/ErrorHandler.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <vector>

#include "compiler/lowering/vectorize/VectorizeComputationsUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace
{

    struct ThGreedyVectorizeComputationsPassRed : public PassWrapper<ThGreedyVectorizeComputationsPassRed, OperationPass<func::FuncOp>> {
        void runOnOperation() final;

        //Function that modifies existing mlir programm structure
        //Currently only gets a pipeline as a list of nodes | cf. Formalisation
        void createVectorizedPipelineOps(func::FuncOp func, std::vector<std::vector<mlir::Operation *>> pipelines) {
            OpBuilder builder(func);
            func->dump();

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
                for(auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
                    auto v = *vIt;

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
            pipelineOp->dump();
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
                        auto matTy = argTy.cast<daphne::MatrixType>();
                        // only remove col information
                        argTy = matTy.withShape(matTy.getNumRows(), -1);
                        break;
                    }
                    case daphne::VectorSplit::GEN:
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
                for (auto z : llvm::zip(v->getResults(), pipelineReplaceResults)) {
                    auto old = std::get<0>(z);
                    auto replacement = std::get<1>(z);

                    // TODO: switch to type based size inference instead
                    // FIXME: if output is dynamic sized, we can't do this
                    // replace `NumRowOp` and `NumColOp`s for output size inference
                    for(auto& use: old.getUses()) {
                        auto* op = use.getOwner();
                        if(auto nrowOp = llvm::dyn_cast<daphne::NumRowsOp>(op)) {
                            nrowOp.replaceAllUsesWith(pipelineOp.getOutRows()[replacement.getResultNumber()]);
                            llvm::outs() << "Replacing ";
                            nrowOp->dump();
                            llvm::outs() << "Replacement: " << pipelineOp.getOutRows()[replacement.getResultNumber()] << "\n";
                            nrowOp.erase();
                        }
                        if(auto ncolOp = llvm::dyn_cast<daphne::NumColsOp>(op)) {
                            ncolOp.replaceAllUsesWith(pipelineOp.getOutCols()[replacement.getResultNumber()]);
                            llvm::outs() << "Replacing ";
                            ncolOp->dump();
                            llvm::outs() << "Replacement: " << pipelineOp.getOutCols()[replacement.getResultNumber()] << "\n";
                            ncolOp.erase();
                        }
                    }
                    // Replace only if not used by pipeline op
                    old.replaceUsesWithIf(
                        replacement, [&](OpOperand &opOperand) {
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

    //Function merges two pipelines into one by appending all operations from one pipeline to another
    //Order is not really considered, as it is embodied in IR
    void mergePipelines(std::vector<std::vector<mlir::Operation*>>& pipelines, std::map<mlir::Operation*, size_t>& operationToPipelineIx, size_t mergeFromIx, size_t mergeIntoIx){
        llvm::outs() << "merge both pipelines\n";
        std::vector<mlir::Operation*> mergedPipeline(pipelines.at(mergeIntoIx));
        for (auto op : pipelines.at(mergeFromIx)) {
            if  (std::find(mergedPipeline.begin(), mergedPipeline.end(), op) == mergedPipeline.end()) {
                mergedPipeline.push_back(op);
                operationToPipelineIx[op] = mergeIntoIx;
            }
        }
        pipelines.at(mergeIntoIx) = std::move(mergedPipeline);
        pipelines.erase(pipelines.begin() + mergeFromIx);
    }


    //-----------------------------------------------------------------
    // Helper Classes
    //-----------------------------------------------------------------

    class HCandidate {
    public:
        HCandidate(mlir::Operation *op1, mlir::Operation *op2) : op1(op1), op2(op2) {}
        mlir::Operation *op1;
        mlir::Operation *op2;
       
        [[maybe_unused]] friend bool operator==(const HCandidate& c1, const HCandidate& c2) {
            return (c1.op1 == c2.op1 && c1.op2 == c2.op2) ||
                (c1.op1 == c2.op2 && c1.op2 == c2.op1);
        }
    };

    class PCCandidate {
    public:
        PCCandidate(mlir::Operation *op1, mlir::Operation *op2) : op1(op1), op2(op2) {}
        mlir::Operation *op1;
        mlir::Operation *op2;

        [[maybe_unused]] friend bool operator==(const PCCandidate& c1, const PCCandidate& c2) {
            return (c1.op1 == c2.op1 && c1.op2 == c2.op2);
        }
    };

    struct DimInfo {
        DimInfo(size_t rows = 0, size_t cols = 0) : rows(rows), cols(cols) {}
        size_t rows;
        size_t cols;
    };

    bool operator==(const DimInfo& d1, const DimInfo& d2) {
        return (d1.rows == d2.rows && d1.cols == d2.cols);
    }
}

namespace std {
    template<>
    struct hash<HCandidate> {
        std::size_t operator()(const HCandidate& c) const {
            //https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
            //careful if c.op1 == c.op2 defaults to 0, for all op
            return hash<mlir::Operation *>()(c.op1) ^ hash<mlir::Operation *>()(c.op2);
        }
    };
    template<>
    struct hash<PCCandidate> {
        std::size_t operator()(const PCCandidate& c) const {
            //https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
            return (hash<mlir::Operation *>()(c.op1) << 1) + hash<mlir::Operation *>()(c.op1) + hash<mlir::Operation *>()(c.op2);
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
    func->walk([&](daphne::Vectorizable op) {
        vectOps.emplace_back(op);
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

    std::unordered_set<PCCandidate> primaryCandidates;
    //TODO: Should this make a weak connection? So in case of not being greedy; first to broken up, if necessary
    std::unordered_set<HCandidate> secondaryCandidates;

    //reversed vectOps
    for (auto opv : vectOps) {

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
            //TODO: Do I need to check whether the operands are even from object type?
            //e.g. what would it mean, if the opv and user shares a constantOp result?
            for (auto user : operand.getUsers()) {
                llvm::outs() << "user: ";
                user->dump();
                
                if (user == opv || //Does not make sense to consider the opv with itself
                    !llvm::dyn_cast<daphne::Vectorizable>(user))//|| //User must be Vectorizable
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
                    secondaryCandidates.insert({opv, user});
                }
            }

            //-----------------------------------------------------------------
            // Producer -> Consumer
            //-----------------------------------------------------------------

            //Get producer of operand
            auto producer = operand.getDefiningOp();
            //Check if producer & consumer are in the same block
            if(llvm::dyn_cast<daphne::Vectorizable>(producer) && (producer->getBlock() == opv->getBlock())) {
                //Currently not needed: checking the split/combine.
                //cf. Algo
                llvm::outs() << "PC-Candidate: " << producer->getName() << " -> " << opv->getName() << "\n";
                primaryCandidates.insert({producer, opv});
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
    for(auto& opv : vectOps) {
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

        size_t opv_pipeIx = opv_it->second;
        llvm::outs() << "opv_pipeId: " << opv_pipeIx << "\n";

        std::vector<decltype(primaryCandidates)::value_type> rel_candidates;
        std::copy_if(primaryCandidates.begin(), primaryCandidates.end(), std::back_inserter(rel_candidates), [opv](const auto& c) {
            return (c.op2 == opv);
        });

        for (auto& candidate : rel_candidates) {

            auto opi_it = operationToPipelineIx.find(candidate.op1);
            llvm::outs() << "opi: " << candidate.op1->getName().getStringRef() << "\n";

            if (opi_it == operationToPipelineIx.end()) {
                pipelines.at(opv_pipeIx).push_back(candidate.op1);
                operationToPipelineIx.insert({candidate.op1, opv_pipeIx});
            }
            else {
                size_t opi_pipeIx = opi_it->second;
                mergePipelines(pipelines, operationToPipelineIx, opi_pipeIx, opv_pipeIx);
            }
        }
        llvm::outs() << "######" << "\n";
    }

    llvm::outs() << "######## END ########" << "\n";

    //Step 4: Horizontal Fusion
    //Separate step as it allows for the producer -> consumer relationship to be exploited first
    //Where does it make a difference?
    llvm::outs() << "######## STEP 4 ########" << "\n";
    for (auto& hcand : secondaryCandidates) {
        
        auto op1_it = operationToPipelineIx.find(hcand.op1);
        auto op2_it = operationToPipelineIx.find(hcand.op2);

        // Check if id is identical, if yes do nothing
        if (op1_it->second == op2_it->second)
            continue;
        
        //TODO: by merging what about the ordering of the operatores inside the fused pipeline?
        //does it matter? same for step 5
        mergePipelines(pipelines, operationToPipelineIx, op2_it->second, op1_it->second);
    }
    llvm::outs() << "######## END ########" << "\n";

    //Step 5: Small pipeline merge, if possible? why not further try to reduce the number of individuals pipelines 
    // and their overhead (e.g. runtime) and merge together if constraints are met (input dimension)
    //Till now we didn´t need to check if dimension matches as they do by definition of considered operators and checked relationship
    //Full potential, if we allow for different output types?

    //careful as it takes the assumption that the size is equal for every object
    //in case of "broadcasting" we need to make it different
    //TODO: check what about SourceOps
#if 1
    llvm::outs() << "######## STEP 5 ########" << "\n";
    auto lambda = [](std::vector<mlir::Operation*> pipeline){
        for (auto op : pipeline) {
            for (auto operandType : op->getOperandTypes()) {
                operandType.dump();
                if (auto opType = llvm::dyn_cast<daphne::MatrixType>(operandType)) {
                    return DimInfo(opType.getNumRows(), opType.getNumCols());
                }
            }
        }
        return DimInfo(0, 0);
    };

    std::vector<std::pair<size_t, DimInfo>> sizes(pipelines.size());
    std::transform(pipelines.begin(), pipelines.end(), sizes.begin(), [lambda](const std::vector<mlir::Operation*>& pipeline) {
        return std::make_pair(pipeline.size(), lambda(pipeline));
    });
    
    for (auto pair : sizes) {
        llvm::outs() << pair.first << " " << pair.second.rows << ":" << pair.second.cols << "\n";
    }

    //dirty
    for (size_t i = 0; i < pipelines.size(); ++i) {
        for (size_t j = i + 1; j < pipelines.size(); ++j) {
            if (lambda(pipelines[i]) == lambda(pipelines[j])) {
                mergePipelines(pipelines, operationToPipelineIx, j, i);
            }
        }
    }
    llvm::outs() << "######## END ########" << "\n";
#endif

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
    //throw std::runtime_error("test");
}


std::unique_ptr<Pass> daphne::createThGreedyVectorizeComputationsPassRed() {
    return std::make_unique<ThGreedyVectorizeComputationsPassRed>();
}
