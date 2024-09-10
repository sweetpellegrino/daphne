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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <util/ErrorHandler.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "compiler/lowering/vectorize/VectorUtils.h"

#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "simdjson.h"

using namespace mlir;

namespace
{

    //-----------------------------------------------------------------
    // Helper Functions
    //-----------------------------------------------------------------

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
    void movePipelineInterleavedOperations(Block::iterator pipelinePosition, const std::vector<mlir::Operation*> &pipeline) {
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

    //-----------------------------------------------------------------
    // Class functions
    //-----------------------------------------------------------------

    struct OneVectorizeComputationsPass : public PassWrapper<OneVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
        const DaphneUserConfig& userConfig;

        explicit OneVectorizeComputationsPass(const DaphneUserConfig& cfg) : userConfig(cfg) {}

        //Function that modifies existing mlir programm structure
        //Currently only gets a pipeline as a list of nodes | cf. Formalisation
        void createVectorizedPipelineOps(func::FuncOp func, std::vector<std::vector<mlir::Operation *>> pipelines, std::map<mlir::Operation*, size_t> decisionIxs) {
            OpBuilder builder(func);

            // Create the `VectorizedPipelineOp`s
            for(auto _pipeline : pipelines) {
                if(_pipeline.empty()) {
                    continue;
                }
                auto valueIsPartOfPipeline = [&](Value operand) {
                    return llvm::any_of(_pipeline, [&](mlir::Operation* lv) { return lv == operand.getDefiningOp(); });
                };
                std::vector<Attribute> vSplitAttrs;
                std::vector<Attribute> vCombineAttrs;
                std::vector<Location> locations;
                std::vector<Value> results;
                std::vector<Value> operands;
                std::vector<Value> outRows;
                std::vector<Value> outCols;

                // first op in pipeline is last in IR
                builder.setInsertionPoint(_pipeline.front());
                // move all operations, between the operations that will be part of the pipeline, before or after the
                // completed pipeline
                movePipelineInterleavedOperations(builder.getInsertionPoint(), _pipeline);

                //potential addition for
                std::vector<mlir::Operation*> pipeline;
                for(auto vIt = _pipeline.rbegin(); vIt != _pipeline.rend(); ++vIt) {
                    auto v = *vIt;

                    auto vSplits = std::vector<daphne::VectorSplit>();
                    auto vCombines = std::vector<daphne::VectorCombine>();
                    auto opsOutputSizes = std::vector<std::pair<Value, Value>>();
                    if (auto vec = llvm::dyn_cast<daphne::Vectorizable>(v)) {
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

                    //check if any of the outputs type of an operator is a scalar value
                    //if yes, add additional castOps inside pipeline and outside pipeline
                    for (size_t i = 0; i < v->getNumResults(); i++) {
                        auto r = v->getResult(0);
                        //TODO: check if it includes all types used in daphne
                        if (r.getType().isIntOrIndexOrFloat()) {
                            auto m1x1 = daphne::MatrixType::get(&getContext(), r.getType(), 1, 1, 1, daphne::MatrixRepresentation::Dense);
                            auto loc = v->getLoc();

                            auto toCastOp = builder.create<daphne::CastOp>(loc, m1x1, r);
                            toCastOp->moveAfter(v);
                            
                            //xxxxxx
                            pipeline.push_back(toCastOp);
                            vCombineAttrs.push_back(daphne::VectorCombineAttr::get(&getContext(), vCombines[i]));
                            auto cst1 = builder.create<daphne::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1l));
                            outRows.push_back(cst1);
                            outCols.push_back(cst1);
                            results.push_back(toCastOp);

                            auto fromCastOp = builder.create<daphne::CastOp>(loc, r.getType(), toCastOp);
                            fromCastOp->moveAfter(toCastOp);
                            r.replaceAllUsesExcept(fromCastOp, toCastOp);
                            
                        }
                    }
                }

                std::vector<Location> locs;
                locs.reserve(_pipeline.size());
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
                        auto matTy = argTy.cast<daphne::MatrixType>();
                        // only remove col information
                        argTy = matTy.withShape(matTy.getNumRows(), -1);
                        break;
                    }
                    case daphne::VectorSplit::NONE:
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
                if (!mlir::sortTopologically(bodyBlock)) {
                    throw std::runtime_error("topoSort");
                }   
            }
        }
    };

    //Function merges two pipelines into one by appending all operations from one pipeline to another
    //Order is not really considered, as it is embodied in IR
    void mergePipelines(std::vector<std::vector<mlir::Operation*>*>& pipelines, std::map<mlir::Operation*, size_t>& operationToPipelineIx, size_t mergeFromIx, size_t mergeIntoIx){
        //llvm::outs() << mergeFromIx << " " << mergeIntoIx << "\n";
        if (mergeFromIx == mergeIntoIx) {
            return;
        }
        if (mergeIntoIx > mergeFromIx) {
            auto temp = mergeFromIx;
            mergeFromIx = mergeIntoIx;
            mergeIntoIx = temp;
        }
        std::vector<mlir::Operation*> *mergedPipeline(pipelines.at(mergeIntoIx));
        for (auto op : *pipelines.at(mergeFromIx)) {
            if  (std::find(mergedPipeline->begin(), mergedPipeline->end(), op) == mergedPipeline->end()) {
                mergedPipeline->push_back(op);
                operationToPipelineIx[op] = mergeIntoIx;
            }
        }
        pipelines.at(mergeIntoIx) = std::move(mergedPipeline);
        pipelines.erase(pipelines.begin() + mergeFromIx);
    }

    std::string getColor(size_t pipelineId) {
        std::vector<std::string> colors = {"tomato", "lightgreen", "lightblue", "plum1", "navajowhite1", "seashell", "hotpink",
                                        "lemonchiffon", "firebrick1", "ivory2", "khaki1", "lightcyan", "olive", "yellow",
                                        "maroon", "violet", "mistyrose2"};
        return colors[pipelineId % colors.size()];
    }

    void printGraph(std::vector<mlir::Operation*> &vectOps, std::vector<size_t> &dIx, std::vector<llvm::SmallVector<int8_t>> &isEdgeActive, std::map<mlir::Operation*, size_t> &operationToPipelineIx, std::string filename) {
        std::ofstream outfile(filename);

        outfile << "digraph G {" << std::endl;

        std::map<mlir::Operation*, std::string> opToNodeName;

        for (size_t i = 0; i < vectOps.size(); ++i) {
            std::string nodeName = "node" + std::to_string(i);
            opToNodeName[vectOps[i]] = nodeName;

            size_t pipelineId = operationToPipelineIx.at(vectOps[i]);
            std::string color = getColor(pipelineId);

            outfile << nodeName << " [label=\"" << vectOps[i]->getName().getStringRef().str() << "\\npIx: " << pipelineId << ", dIx: " << dIx[i] << "\", fillcolor=" << color << ", style=filled];" << std::endl;
        }

        for (size_t i = 0; i < vectOps.size(); ++i) {
            const auto &edges = isEdgeActive[i];
            mlir::Operation* op = vectOps[i];

            for (size_t j = 0; j < op->getNumOperands(); ++j) {
                if (edges[j] == 1 || edges[j] == 0) {
                    mlir::Operation* operandOp = op->getOperand(j).getDefiningOp();

                    outfile << opToNodeName[operandOp] << " -> " << opToNodeName[op];

                    if (edges[j] == 0) {
                        outfile << " [style=dotted]";
                    }

                    outfile << ";" << std::endl;
                }
            }
        }
        outfile << "}" << std::endl;
    }

    void readFromJsonElement(std::string elementKey, std::vector<size_t>& dIx, std::vector<llvm::SmallVector<int8_t>>& isEdgeActive) {
        
        simdjson::ondemand::parser parser;
        simdjson::padded_string json = simdjson::padded_string::load("data.json");
        simdjson::ondemand::document tweets = parser.iterate(json);

        auto _value = tweets[elementKey].get_object();
        for (simdjson::ondemand::value value : _value["dIx"].get_array()) {
            dIx.push_back(value.get_uint64());
        }

        for (simdjson::ondemand::array arr : _value["isEdgeActive"].get_array()) {
            llvm::SmallVector<int8_t> sm;
            for (simdjson::ondemand::value value : arr) {
                sm.push_back(static_cast<int8_t>(value.get_int64()));
            }
            isEdgeActive.push_back(sm);
        }
    }
}

void OneVectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    spdlog::debug("OneVectorizeComputationsPass");

    std::vector<mlir::Operation *> vectOps;
    func->walk([&](daphne::Vectorizable op) {
        vectOps.emplace_back(op);
    });
    std::reverse(vectOps.begin(), vectOps.end());

    std::string key = std::to_string(userConfig.runCombKey);

    std::vector<size_t> dIx;
    std::vector<llvm::SmallVector<int8_t>> isEdgeActive;

    readFromJsonElement(key, dIx, isEdgeActive);
    
    //generate pipelines from isEdgeActive

    std::vector<std::vector<mlir::Operation*>*> pipelines;
    std::map<mlir::Operation*, size_t> operationToPipelineIx;
    
    llvm::outs() << "..................................................................";
    for (llvm::SmallVector<int8_t> smallVector : isEdgeActive) {
        llvm::outs() << "(";
        for (int8_t value : smallVector) {
            llvm::outs() << static_cast<int>(value) << ", ";
        }
        llvm::outs() << "), ";
    }
    llvm::outs() << "\n";

    //(0, ), (0, 0, ), (-1, ), (0, 0, ), (-1, ), (-1, ), 

    for (size_t i = 0; i < isEdgeActive.size(); i++) {
        auto e = isEdgeActive.at(i);
        auto v = vectOps.at(i);

        //check if already in pipeline
        size_t pipelineIx = 0;
        std::vector<mlir::Operation*> *pipeline;
        if (operationToPipelineIx.find(v) == operationToPipelineIx.end()) {
            pipelineIx = pipelines.size();
            operationToPipelineIx.insert({v, pipelines.size()});
            pipeline = new std::vector<mlir::Operation*>();
            pipeline->push_back(v);
            pipelines.push_back(pipeline);
        }
        else {
            pipelineIx = operationToPipelineIx.at(v);
            pipeline = pipelines.at(pipelineIx);
        }

        for (size_t j = 0; j < e.size(); j++)
            if (e[j] == 1) {
                auto x = v->getOperand(j).getDefiningOp();
                //x->print(llvm::outs());
                //llvm::outs() << "\n";
                if (operationToPipelineIx.find(x) == operationToPipelineIx.end()) {
                    operationToPipelineIx.insert({x, pipelineIx});
                    pipeline->push_back(x);
                }
                else {
                    mergePipelines(pipelines, operationToPipelineIx, pipelineIx, operationToPipelineIx.at(x));
                    pipelineIx = operationToPipelineIx.at(x);
                    pipeline = pipelines.at(pipelineIx);
                }
            }
        
    }


    //-----------------------------------------------------------------

    std::vector<std::vector<mlir::Operation*>> _pipelines;
    _pipelines.resize(pipelines.size());

    std::transform(pipelines.begin(), pipelines.end(), _pipelines.begin(), [](const auto& ptr) { return *ptr; }); 

    std::map<mlir::Operation*, size_t> decisionIx;
    for(size_t i = 0; i < dIx.size(); i++) {
        decisionIx.insert({vectOps.at(i), dIx.at(i)});
    }

    std::string filename = "graphs/graph-" + key + ".dot";
    //printGraph(vectOps, dIx, isEdgeActive, operationToPipelineIx, filename);
    VectorUtils::printPipelines(vectOps, operationToPipelineIx, dIx, filename);

    OneVectorizeComputationsPass::createVectorizedPipelineOps(func, _pipelines, decisionIx);

 
    return;
}


std::unique_ptr<Pass> daphne::createOneVectorizeComputationsPass(const DaphneUserConfig& cfg) {
    return std::make_unique<OneVectorizeComputationsPass>(cfg);
}