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

        void createVectorizedPipelineOps(mlir::func::FuncOp func, std::vector<Pipeline> pipelines, std::map<mlir::Operation*, VectorIndex> decisionIxs) {
            mlir::OpBuilder builder(func);

            llvm::outs() << "test2" << pipelines.size() << "\n";

            for (auto pipe : pipelines) {
                for (auto op : pipe) {
                    op->print(llvm::outs());
                    llvm::outs() << "\n";
                }
            }

            // Create the `VectorizedPipelineOp`s
            for(auto _pipeline : pipelines) {
                llvm::outs() << "testp: " << _pipeline.size() << "\n";
                if(_pipeline.empty())
                    continue;
                
                auto valueIsPartOfPipeline = [&](mlir::Value operand) {
                    return llvm::any_of(_pipeline, [&](mlir::Operation* lv) { return lv == operand.getDefiningOp(); });
                };
                llvm::outs() << "test3: " << _pipeline.size() << "\n";
                std::vector<mlir::Attribute> vSplitAttrs;
                std::vector<mlir::Attribute> vCombineAttrs;
                std::vector<mlir::Location> locations;
                std::vector<mlir::Value> results;
                std::vector<mlir::Value> operands;
                std::vector<mlir::Value> outRows;
                std::vector<mlir::Value> outCols;

                // first op in pipeline is last in IR
                builder.setInsertionPoint(_pipeline.front());
                // move all operations, between the operations that will be part of the pipeline, before or after the
                // completed pipeline
                for (auto op : _pipeline) {
                    op->print(llvm::outs());
                    llvm::outs() << "\n";
                }
                VectorUtils::movePipelineInterleavedOperations(builder.getInsertionPoint(), _pipeline);

                

                //potential addition for
                std::vector<mlir::Operation*> pipeline;
                llvm::outs() << _pipeline.size() << "\n";
                for(auto vIt = _pipeline.rbegin(); vIt != _pipeline.rend(); ++vIt) {
                    auto v = *vIt;

                    v->print(llvm::outs());
                    llvm::outs() << "\n";

                    auto vSplits = std::vector<mlir::daphne::VectorSplit>();
                    auto vCombines = std::vector<mlir::daphne::VectorCombine>();
                    auto opsOutputSizes = std::vector<std::pair<mlir::Value, mlir::Value>>();
                    if (auto vec = llvm::dyn_cast<mlir::daphne::Vectorizable>(v)) {
                        //vec->print(llvm::outs());
                        //llvm::outs() << "\n";
                        size_t d = decisionIxs[v];
                        vSplits = vec.getVectorSplits()[d];
                        vCombines = vec.getVectorCombines()[d];
                        opsOutputSizes = vec.createOpsOutputSizes(builder)[d];
                    } else {
                        throw std::runtime_error("Vectorizable op not found");
                    }

                    pipeline.push_back(v);

                    // TODO: although we do create enum attributes, it might make sense/make it easier to
                    // just directly use an I64ArrayAttribute
                    // Determination of operands of VectorizedPipelineOps!
                    for(auto i = 0u; i < v->getNumOperands(); ++i) {
                        auto operand = v->getOperand(i);
                        if(!valueIsPartOfPipeline(operand)){
                            vSplitAttrs.push_back(mlir::daphne::VectorSplitAttr::get(&getContext(), vSplits[i]));
                            operands.push_back(operand);
                        }
                    }

                    // Determination of results of VectorizedPipelineOps!
                    for(auto vCombine : vCombines) {
                        vCombineAttrs.push_back(mlir::daphne::VectorCombineAttr::get(&getContext(), vCombine));
                    }
                    locations.push_back(v->getLoc());
                    for(auto result: v->getResults()) {
                        results.push_back(result);
                    }
                    for(auto outSize: opsOutputSizes) {
                        outRows.push_back(outSize.first);
                        outCols.push_back(outSize.second);
                    }

                    //check if any of the outputs type of an operator is a scalar value
                    //if yes, add additional castOps inside pipeline and outside pipeline
                    for (size_t i = 0; i < v->getNumResults(); i++) {
                        auto r = v->getResult(0);
                        //TODO: check if it includes all types used in daphne
                        if (r.getType().isIntOrIndexOrFloat()) {
                            auto m1x1 = mlir::daphne::MatrixType::get(&getContext(), r.getType(), 1, 1, 1, mlir::daphne::MatrixRepresentation::Dense);
                            auto loc = v->getLoc();

                            auto toCastOp = builder.create<mlir::daphne::CastOp>(loc, m1x1, r);
                            toCastOp->moveAfter(v);
                            
                            //xxxxxx
                            pipeline.push_back(toCastOp);
                            vCombineAttrs.push_back(mlir::daphne::VectorCombineAttr::get(&getContext(), vCombines[i]));
                            auto cst1 = builder.create<mlir::daphne::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1l));
                            outRows.push_back(cst1);
                            outCols.push_back(cst1);
                            results.push_back(toCastOp);

                            auto fromCastOp = builder.create<mlir::daphne::CastOp>(loc, r.getType(), toCastOp);
                            r.replaceAllUsesExcept(fromCastOp, toCastOp);

                            //
                            mlir::Operation* firstUseOp;
                            for (const auto &use : fromCastOp->getUses()) {
                                firstUseOp = use.getOwner(); 
                                break;
                            }
                            fromCastOp->moveBefore(firstUseOp);
                            
                        }
                    }
                }

                std::vector<mlir::Location> locs;
                locs.reserve(_pipeline.size());
                for(auto op: pipeline) {
                    locs.push_back(op->getLoc());
                }

                auto loc = builder.getFusedLoc(locs);
                auto pipelineOp = builder.create<mlir::daphne::VectorizedPipelineOp>(loc,
                    mlir::ValueRange(results).getTypes(),
                    operands,
                    outRows,
                    outCols,
                    builder.getArrayAttr(vSplitAttrs),
                    builder.getArrayAttr(vCombineAttrs),
                    nullptr);
                mlir::Block *bodyBlock = builder.createBlock(&pipelineOp.getBody());

                //remove information from input matrices of pipeline
                for(size_t i = 0u; i < operands.size(); ++i) {
                    auto argTy = operands[i].getType();
                    switch (vSplitAttrs[i].cast<mlir::daphne::VectorSplitAttr>().getValue()) {
                        case mlir::daphne::VectorSplit::ROWS: {
                            auto matTy = argTy.cast<mlir::daphne::MatrixType>();
                            // only remove row information
                            argTy = matTy.withShape(-1, matTy.getNumCols());
                            break;
                        }
                        case mlir::daphne::VectorSplit::COLS: {
                            auto matTy = argTy.cast<mlir::daphne::MatrixType>();
                            // only remove col information
                            argTy = matTy.withShape(matTy.getNumRows(), -1);
                            break;
                        }
                        case mlir::daphne::VectorSplit::NONE:
                            // keep any size information
                            break;
                    }
                    bodyBlock->addArgument(argTy, builder.getUnknownLoc());
                }

                auto argsIx = 0u;
                auto resultsIx = 0u;
                //for every op in pipeline
                for(auto vIt = pipeline.begin(); vIt != pipeline.end(); ++vIt) {
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
                            if(auto nrowOp = llvm::dyn_cast<mlir::daphne::NumRowsOp>(op)) {
                                nrowOp.replaceAllUsesWith(pipelineOp.getOutRows()[replacement.getResultNumber()]);
                                nrowOp.erase();
                            }
                            if(auto ncolOp = llvm::dyn_cast<mlir::daphne::NumColsOp>(op)) {
                                ncolOp.replaceAllUsesWith(pipelineOp.getOutCols()[replacement.getResultNumber()]);
                                ncolOp.erase();
                            }
                        }
                        // Replace only if not used by pipeline op
                        old.replaceUsesWithIf(
                            replacement, [&](mlir::OpOperand &opOperand) {
                                return llvm::count(pipeline, opOperand.getOwner()) == 0;
                            });
                        }
                }
                bodyBlock->walk([](mlir::Operation* op) {
                    for(auto resVal: op->getResults()) {
                        if(auto ty = resVal.getType().dyn_cast<mlir::daphne::MatrixType>()) {
                            resVal.setType(ty.withShape(-1, -1));
                        }
                    }
                });
                builder.setInsertionPointToEnd(bodyBlock);
                builder.create<mlir::daphne::ReturnOp>(loc, results);
                if (!mlir::sortTopologically(bodyBlock)) {
                    throw std::runtime_error("topoSort");
                }   
            }
        }
    }; 

    /*
    class Pipeline {
    private: 
        std::vector<mlir::Operation*> operations;
        std::vector<mlir::Operation*> inputOperations;
        std::vector<mlir::Operation*> outputOperations;

        std::map<mlir::Operation*, VectorIndex> decisions;

    public:
        void addOperation(mlir::Operation* op, VectorIndex vectIx) {
            operations.push_back(op);
            decisions.insert({op, vectIx});
        }

        void updateInputOutputOperations(std::vector<mlir::Operation*> ops) {

            std::vector<mlir::Operation*> _io;
            std::vector<mlir::Operation*> _oo;

            for (auto op : operations) {
                bool isInput = true;
                bool isOutput = true;

                for (const auto operand : op->getOperands()) {
                    mlir::Operation* defOp = operand.getDefiningOp(); 

                    //check if feasible for blockargs
                    if (!defOp)
                        continue;
                    else {
                        if (std::find(ops.begin(), ops.end(), defOp) == ops.end()) {
                            isInput = false;
                            break;
                        }
                    }
                }

                for (const auto user : op->getUsers()) {
                    if (std::find(ops.begin(), ops.end(), user) == ops.end()) {
                        isOutput = false;
                        break;
                    }
                }

                if (isInput)
                    _io.push_back(op);
                if (isOutput)
                    _oo.push_back(op);
            }

            inputOperations = _io;
            outputOperations = _oo;
        }

        VectorIndex getVectorIndexOfOperation(mlir::Operation* op) {
            return decisions.at(op);
        }

        void print() {
            llvm::outs() << "Pipeline:\n";
            llvm::outs() << "Ops: ";
            for (const auto op : operations) {
                llvm::outs() << "(" << op->getName().getStringRef().str() << ", vIx: " << decisions.at(op) << "), ";
            }
            llvm::outs() << "\n";

            llvm::outs() << "Inputs: ";
            for (const auto op : inputOperations) {
                llvm::outs() << op->getName().getStringRef().str() << ", ";
            }
            llvm::outs() << "\n";
            llvm::outs() << "Outputs: ";
            for (const auto op : outputOperations) {
                llvm::outs() << op->getName().getStringRef().str() << ", ";
            }
            llvm::outs() << "\n";
        }

    };
    */

 

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

    llvm::outs() << "Greedy1VectorizeComputationsPass" << "\n";

    std::vector<mlir::Operation*> ops;
    func->walk([&](daphne::Vectorizable op) {
        ops.emplace_back(op);
    });
    std::reverse(ops.begin(), ops.end()); 

    for (auto op : ops) {
        op->print(llvm::outs());
        llvm::outs() << "\n";
    }
    
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

    llvm::outs() << "-------------------------LEAFOPS----------------------------" << "\n";

    for (auto op : leafOps) {
        op->print(llvm::outs());
        llvm::outs() << "\n";
    }

    VectorUtils::DEBUG::printGraph(leafOps, "graph-gr1-pre.dot");

    llvm::outs() << "----------------------------------------------------------" << "\n";

    std::multimap<PipelinePair, DisconnectReason> producerConsumerRelationship;
    std::map<mlir::Operation*, Pipeline*> operationToPipeline;

    //int nextPipelineId;

    while (!stack.empty()) {
        auto t = stack.top(); stack.pop();
        auto op = std::get<0>(t);
        auto currPipeline = std::get<1>(t);
        auto disReason = std::get<2>(t);

        if(operationToPipeline.find(op) != operationToPipeline.end()) {
            auto producerPipeline = operationToPipeline.at(op);
            producerConsumerRelationship.insert({{producerPipeline, currPipeline}, disReason});
            continue;
        }

        if (disReason != DisconnectReason::NONE) {
            auto _pipeline = new Pipeline();
            pipelines.push_back(_pipeline);
            
            //check needed for empty init
            if (currPipeline != nullptr)
                producerConsumerRelationship.insert({{_pipeline, currPipeline}, disReason});

            currPipeline = _pipeline;
        }

        operationToPipeline.insert({op, currPipeline});
        currPipeline->push_back(op);

        auto vectOp = llvm::dyn_cast<daphne::Vectorizable>(op);

        for (size_t i = 0; i < vectOp->getNumOperands(); ++i) {
            auto operand = vectOp->getOperand(i);

            if (!llvm::isa<daphne::MatrixType>(operand.getType()))
                continue;

            llvm::outs() << vectOp->getName().getStringRef().str() << " ";

            if (llvm::isa<mlir::BlockArgument>(operand)) {
                llvm::outs() << "Block" << "\n";
                continue;
            }

            if (auto vectDefOp = llvm::dyn_cast<daphne::Vectorizable>(operand.getDefiningOp())) {
                llvm::outs() << vectDefOp->getName().getStringRef().str() << "\n";

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
            }else {
                llvm::outs() << "\n";
            }
        }
        //printStack(stack);

    }

    std::map<mlir::Operation*, size_t> decisionIxs;
    for (const auto& op : ops) {
        decisionIxs.insert({op, 0});
    }
    
    VectorUtils::DEBUG::printPipelines(ops, operationToPipeline, decisionIxs, "graph-gr1-post.dot");

    //post processing
    std::map<PipelinePair, DisconnectReason> _producerConsumerRelationship;
    for (const auto& [key, value] : producerConsumerRelationship) {
        if (_producerConsumerRelationship.find(key) == _producerConsumerRelationship.end()) {
            _producerConsumerRelationship.insert({key, value});
        }
        else {
            //only overwrite if invalid as it domiantes the multi-consumer
            if(value == DisconnectReason::INVALID) {
                _producerConsumerRelationship.insert_or_assign(key, value);
            }
        }
    }

    for (auto p : pipelines) {
        llvm::outs() << p << " ";
        for (auto op : *p) {
            llvm::outs() << op->getName().getStringRef().str() << ", ";
        }
        llvm::outs() << "\n";
    }

    for (const auto& [key, value]  : _producerConsumerRelationship) {
        llvm::outs() << "(" << key.first << ", " << key.second << "|" << int(value) << ")\n";
    }

    llvm::outs() << "----------------------------------------------------------" << "\n";
    
    bool change = true;
    while (change) {
        change = false;
        
        llvm::outs() << _producerConsumerRelationship.size() << "\n";

        std::multimap<PipelinePair, DisconnectReason> pcr;
        for (const auto& [rel, disReason] : _producerConsumerRelationship) {

            llvm::outs() << VectorUtils::DEBUG::printPtr(rel.first) << " " << VectorUtils::DEBUG::printPtr(rel.second) << " " << int(disReason) << "\n";

            if (disReason == DisconnectReason::INVALID)
                continue;

            if (VectorUtils::tryTopologicalSortMerged(pipelines, _producerConsumerRelationship, rel.first, rel.second)) {
                llvm::outs() << "true" << "\n";
                //merge
                auto mergedPipeline = VectorUtils::mergePipelines(pipelines, operationToPipeline, rel.first, rel.second);
                
                llvm::outs() << VectorUtils::DEBUG::printPtr(mergedPipeline) << "\n";

                //_producerConsumerRelationship.erase(rel);

                for (const auto& [_rel, _disReason] : _producerConsumerRelationship) {
                    //error::::::::::::
                    //error::::::::::::
                    //error::::::::::::
                    llvm::outs() << "\n";
                    llvm::outs() << VectorUtils::DEBUG::printPtr(_rel.first) << " " << VectorUtils::DEBUG::printPtr(_rel.second) << " " << VectorUtils::DEBUG::printPtr(rel.first) << " " << VectorUtils::DEBUG::printPtr(rel.second) << "\n";

                    if(_rel.first == rel.first && _rel.second == rel.second)
                        continue;

                    if (_rel.first == rel.first || _rel.first == rel.second) {
                        auto newRel = std::make_pair(mergedPipeline, _rel.second);
                        llvm::outs() << "1: " << VectorUtils::DEBUG::printPtr(newRel.first) << " " << VectorUtils::DEBUG::printPtr(newRel.second) << " " << int(_disReason) << "\n";
                        pcr.insert({newRel, _disReason});
                    }
                    else if (_rel.second == rel.first || _rel.second == rel.second) {
                        auto newRel = std::make_pair(_rel.first, mergedPipeline);
                        llvm::outs() << "2: " << VectorUtils::DEBUG::printPtr(newRel.first) << " " << VectorUtils::DEBUG::printPtr(newRel.second) << " " << int(_disReason) << "\n";
                        pcr.insert({newRel, _disReason});
                    }
                    else { 
                        llvm::outs() << "3: " << VectorUtils::DEBUG::printPtr(_rel.first) << " " << VectorUtils::DEBUG::printPtr(_rel.second) << " " << int(_disReason) << "\n";
                        pcr.insert({_rel, _disReason});
                    }
                
                }

                llvm::outs() << "\n";
                for (const auto& [key, value]  : _producerConsumerRelationship) {
                    llvm::outs() << "(" << VectorUtils::DEBUG::printPtr(key.first) << ", " << VectorUtils::DEBUG::printPtr(key.second) << "|" << int(value) << ")\n";
                } 
                change = true;
                break;
            }
        }

        std::map<PipelinePair, DisconnectReason> _pcr;
        for (const auto& [key, value] : pcr) {
            if (_pcr.find(key) == _pcr.end()) {
                _pcr.insert({key, value});
            }
            else {
                //only overwrite if invalid as it domiantes the multi-consumer
                if(value == DisconnectReason::INVALID) {
                    _pcr.insert_or_assign(key, value);
                }
            }
        }

        _producerConsumerRelationship = _pcr;

        llvm::outs() << "\n";
        for (const auto& [key, value]  : _producerConsumerRelationship) {
            llvm::outs() << "(" << VectorUtils::DEBUG::printPtr(key.first) << ", " << VectorUtils::DEBUG::printPtr(key.second) << "|" << int(value) << ")\n";
        } 

        for (auto p : pipelines) {
            llvm::outs() << VectorUtils::DEBUG::printPtr(p) << " ";
            for (auto op : *p) {
                llvm::outs() << op->getName().getStringRef().str() << ", ";
            }
            llvm::outs() << "\n";
        }
        llvm::outs() << "----------------------------------------------------------" << "\n";
    }

    llvm::outs() << "----------------------------------------------------------" << "\n";

    for (const auto& [key, value]  : _producerConsumerRelationship) {
        llvm::outs() << "(" << key.first << ", " << key.second << "|" << int(value) << ")\n";
    } 

    llvm::outs() << "----------------------------------------------------------" << "\n";

    for (auto p : pipelines) {
        llvm::outs() << VectorUtils::DEBUG::printPtr(p) << " ";
        for (auto op : *p) {
            llvm::outs() << op->getName().getStringRef().str() << ", ";
        }
        llvm::outs() << "\n";
    }

    VectorUtils::DEBUG::printPipelines(ops, operationToPipeline, decisionIxs, "graph-gr1.dot");

    std::vector<Pipeline> _pipelines;
    _pipelines.resize(pipelines.size());

    //std::transform(pipelines.begin(), pipelines.end(), _pipelines.begin(), [](const auto& ptr) { return *ptr; }); 

    for (auto pipe : pipelines) {
        Pipeline _p;
        for (auto op : *pipe) {
            op->print(llvm::outs());
            llvm::outs() << "\n";
            _p.push_back(op);
        }
        _pipelines.push_back(_p);
    }

    llvm::outs() << "test1" << "\n";

    Greedy1VectorizeComputationsPass::createVectorizedPipelineOps(func, _pipelines, decisionIxs);

    llvm::outs() << "test" << "\n";

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