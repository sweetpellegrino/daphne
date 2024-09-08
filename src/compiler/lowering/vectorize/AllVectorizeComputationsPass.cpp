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
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <util/ErrorHandler.h>
#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Passes.h"
#include <stack>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace
{

    //-----------------------------------------------------------------
    // Helper Classes
    //-----------------------------------------------------------------
    
    class Candidate {
    public:
        Candidate(mlir::Operation *op1, mlir::Operation *op2) : op1(op1), op2(op2) {}
        mlir::Operation *op1;
        mlir::Operation *op2; 
    };

    class HCandidate : public Candidate {
    public:
        HCandidate(mlir::Operation *op1, mlir::Operation *op2) : Candidate(op1, op2){}
        [[maybe_unused]] friend bool operator==(const HCandidate& c1, const HCandidate& c2) {
            return (c1.op1 == c2.op1 && c1.op2 == c2.op2) ||
                (c1.op1 == c2.op2 && c1.op2 == c2.op1);
        }
    };

    class PCCandidate : public Candidate {
    public:
        PCCandidate(mlir::Operation *op1, mlir::Operation *op2) : Candidate(op1, op2){}
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

    struct AllVectorizeComputationsPass : public PassWrapper<AllVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;

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

            llvm::outs() << "####replace####\n";
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
                llvm::outs() << "####end####\n";
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

    void printStack(std::stack<mlir::Operation*> stack) {
        std::stack<mlir::Operation*> temp = stack; 

        llvm::outs() << "### Stack ###" << "\n";
        while (!temp.empty()) {
            mlir::Operation* op = temp.top();
            op->print(llvm::outs());
            llvm::outs() << "\n";
            temp.pop();
        }
        llvm::outs() << "### stack ###" << "\n";
    }

    void printGraph(mlir::Operation* op, std::string filename) {
        std::stack<mlir::Operation*> stack;
        std::ofstream dot(filename);
        if (!dot.is_open()) {
            throw std::runtime_error("test");
        }

        dot << "digraph G {\n";
        stack.push(op);

        std::vector<mlir::Operation*> visited;

        while (!stack.empty()) {
            op = stack.top(); stack.pop();
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

    void printGraph(std::vector<mlir::Operation*> startOps, std::string filename) {
        std::stack<mlir::Operation*> stack;
        std::ofstream dot(filename);
        if (!dot.is_open()) {
            throw std::runtime_error("test");
        }

        dot << "digraph G {\n";
        for (auto s : startOps) {
            stack.push(s);
            stack.push(s);
        }

        std::vector<mlir::Operation*> visited;

        while (!stack.empty()) {
            auto op = stack.top(); stack.pop();
            if(std::find(visited.begin(), visited.end(), op) != visited.end()) {
                continue;
            }
            visited.push_back(op);

            auto v = llvm::dyn_cast<daphne::Vectorizable>(op);
            if (!v) 
                continue;

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

    //void backward_propagation(mlir::Operation* op, std::map<mlir::Operation*, bool> visited) {
    /*void backward_propagation(mlir::Operation* op, std::vector<mlir::Operation*> *visited, daphne::VectorSplit* expected_split) {
        //check if operation already in visited?
        if(std::find(visited->begin(), visited->end(), op) != visited->end()) {
            //already visited
            return; 
        }
        visited->push_back(op);
        op->print(llvm::outs());
        llvm::outs() << "\n";
        auto v = llvm::dyn_cast<daphne::Vectorizable>(op);
        //check compability
        std::vector<daphne::VectorSplit> v_splits = v.getVectorSplits()[0];
        if (expected_split == nullptr) {
            expected_split = &v_splits[0]; 
        }
        for(auto e : llvm::zip(v->getOperands(),v_splits)) {
            daphne::VectorSplit split = std::get<1>(e);
            if (llvm::isa<daphne::EwAddOp>(op)) {
                if(split != *expected_split) {
                    throw std::runtime_error("collision");
                } 
            }
            auto defOp = std::get<0>(e).getDefiningOp();
            //careful with reduction ops
            if (llvm::isa<daphne::MatrixType>(std::get<0>(e).getType()) && llvm::isa<daphne::Vectorizable>(defOp)) { 
                backward_propagation(defOp, visited, expected_split);
            }
        }
    }*/

    std::map<mlir::Operation*, size_t> backward_propagation(mlir::Operation* op) {

        std::stack<std::pair<mlir::Operation*, size_t>> stack;
        std::unordered_set<mlir::Operation*> visited;
        std::map<mlir::Operation*, size_t> decisionIxs;

        stack.push({op, 0});

        while (!stack.empty()) {
            auto t = stack.top(); stack.pop();
            mlir::Operation* op = t.first;
            size_t d = t.second;

            if(std::find(visited.begin(), visited.end(), op) != visited.end())
                continue;

            auto v = llvm::dyn_cast<daphne::Vectorizable>(op);

            visited.insert(op);
            decisionIxs[op] = d;

            for (size_t i = 0; i < v->getNumOperands(); ++i) {
                auto operand = v->getOperand(i);

                if (!llvm::isa<daphne::MatrixType>(operand.getType())) 
                    continue;

                if(auto v_defOp = llvm::dyn_cast<daphne::Vectorizable>(operand.getDefiningOp())) {
                    auto v_operandSplit = v.getVectorSplits()[d][i];

                    for (size_t j = 0; j < v_defOp.getVectorCombines().size(); ++j) {
                        auto v_defOp_operandCombine  = v_defOp.getVectorCombines()[j];

                        daphne::VectorCombine _operandCombine;
                        switch (v_operandSplit) {
                            case daphne::VectorSplit::ROWS:
                                _operandCombine = daphne::VectorCombine::ROWS;
                                break;
                            case daphne::VectorSplit::COLS:
                                _operandCombine = daphne::VectorCombine::COLS;
                                break;
                            //would be the reason to split a pipeline!
                            /*case daphne::VectorSplit::NONE:
                                 _operandCombine = daphne::VectorCombine::NONE;
                                break*/
                            default:
                                throw std::runtime_error("?????");
                        }
                        //only supporting a single return of an operation, cf. index 0
                        if (v_defOp_operandCombine[0] == _operandCombine) {
                            llvm::outs() << "push stack: " << v_defOp->getName() << ", j=" << j << "\n";
                            stack.push({v_defOp, j});
                        }
                    }
                } 
            }
        }
        return decisionIxs;
    }

    bool matchingVectorSplitCombine(const daphne::VectorSplit split, const daphne::VectorCombine combine) {
        daphne::VectorCombine _operandCombine;
        switch (split) {
            case daphne::VectorSplit::ROWS:
                _operandCombine = daphne::VectorCombine::ROWS;
                break;
            case daphne::VectorSplit::COLS:
                _operandCombine = daphne::VectorCombine::COLS;
                break;
            default:
                //No matching split/combine; basically resulting in separate pipelines
                return false;
        }
        if (combine == _operandCombine)
            return true;
        return false;
    }

    void generate_decisionIxs_combinations(std::vector<std::vector<size_t>> &combinations, const std::vector<mlir::Operation *> &vectOps, std::vector<size_t> _combination, size_t vectIx) {
        if (vectIx == vectOps.size()) {
            combinations.push_back(_combination);
            return; 
        }

        auto op = llvm::dyn_cast<daphne::Vectorizable>(vectOps.at(vectIx));
        for (size_t i = 0; i < op.getVectorSplits().size(); i++) {
            _combination.push_back(i);
            generate_decisionIxs_combinations(combinations, vectOps, _combination, vectIx + 1);
            _combination.pop_back();
        }

    }

    void generate_isEdgeActivated_combinations(std::vector<std::vector<llvm::SmallVector<int8_t>>> &combinations, const std::vector<mlir::Operation *> &vectOps, std::vector<llvm::SmallVector<int8_t>> _combination, llvm::SmallVector<int8_t> _operands, size_t vectIx, size_t operandIx) {
      
        if (vectOps.at(vectIx)->getNumOperands() == operandIx) {
            _combination.push_back(_operands);
            _operands = llvm::SmallVector<int8_t>();
            vectIx++;
            operandIx = 0;
        }  

        if (vectIx == vectOps.size()) {
            combinations.push_back(_combination);
            return; 
        }

        auto defOp = vectOps.at(vectIx)->getOperand(operandIx).getDefiningOp();
        
        if (std::find(vectOps.begin(), vectOps.end(), defOp) == vectOps.end()) { //block?) {
            _operands.push_back(-1);
            generate_isEdgeActivated_combinations(combinations, vectOps, _combination, _operands, vectIx, operandIx + 1);
            _operands.pop_back();
        } else {
            _operands.push_back(0);
            generate_isEdgeActivated_combinations(combinations, vectOps, _combination, _operands, vectIx, operandIx + 1);
            _operands.pop_back();

            _operands.push_back(1);
            generate_isEdgeActivated_combinations(combinations, vectOps, _combination, _operands, vectIx, operandIx + 1);
            _operands.pop_back();
        }
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

    
void AllVectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    llvm::outs() << "AllVectorizeComputationsPass" << "\n";

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
        op->print(llvm::outs());
        llvm::outs() << "\n";
    }

    llvm::outs() << "######## END ########" << "\n";

    //find starting ops
    std::vector<mlir::Operation*> startOps;
    for (auto op : vectOps) {
        auto users = op->getUsers();
        bool found = false;
        for (auto u :users) {
            if (std::find(vectOps.begin(), vectOps.end(), u) != vectOps.end()) { 
                found = true;
                break;
            }
        }
        if(!found)
            startOps.push_back(op);
    }
    
    printGraph(startOps, "graph.dot"); 

    //Improvment: can we already know that some operations can not be fused???
    //Step 2: Identify merging candidates
    llvm::outs() << "######## STEP 2 ########" << "\n";

    //estimated combinations
    size_t v = vectOps.size();

    size_t e = 0;
    for (auto &op : vectOps) {
        for (auto t : op->getOperands()) {

            if(op->getBlock() != t.getDefiningOp()->getBlock()) {
                continue;
            }

            if (std::find(vectOps.begin(), vectOps.end(), t.getDefiningOp()) != vectOps.end()) {
                e++;
            }
        }
    }

    int64_t f = (v * std::pow(2, e));
    llvm::outs() << "v: " << v << ", " << "e:" << e << "\n";
    llvm::outs() << "Estimated: " << f << "\n";

    std::vector<std::vector<size_t>> dIx_combinations;
    generate_decisionIxs_combinations(dIx_combinations, vectOps, {}, 0);

    llvm::outs() << "dIx_combinations size: " << dIx_combinations.size() << "\n";

#if 0
    for (auto combination : combinations) {
        for (size_t index : combination) {
            llvm::outs() << index << " ";
        }
        llvm::outs() << "\n";
    }
#endif

    std::vector<std::vector<llvm::SmallVector<int8_t>>> isEdgeActivated_combinations;
    std::vector<llvm::SmallVector<int8_t>> _isEdgeActivated;

    generate_isEdgeActivated_combinations(isEdgeActivated_combinations, vectOps, _isEdgeActivated, {}, 0, 0);

    llvm::outs() << "isEdgeActivated_combinations size: " << isEdgeActivated_combinations.size() << "\n";

#if 0

    for (auto combination : isEdgeActivated_combinations) {
        for (llvm::SmallVector<int8_t> smallVector : combination) {
            llvm::outs() << "(";
            for (int8_t value : smallVector) {
                llvm::outs() << static_cast<int>(value) << ", ";
            }
            llvm::outs() << "), ";
        }
        llvm::outs() << "\n";
    }

#endif

    llvm::outs() << "----------------------------------------------------------\n";

    std::vector<std::map<mlir::Operation*, size_t>> dIxs;

    for (auto d : dIx_combinations) {

        std::map<mlir::Operation*, size_t> decisionIx;
        for(size_t i = 0; i < d.size(); i++) {
            decisionIx.insert({vectOps.at(i), d.at(i)});
        }
        dIxs.push_back(decisionIx);
    }

    llvm::outs() << "dIxs: " << dIxs.size() << "\n";

    std::vector<std::vector<std::vector<mlir::Operation*>*>> pipeline_groups;
    std::vector<std::map<mlir::Operation*, size_t>> pipeline_groups_map;

    for (auto edges : isEdgeActivated_combinations) {

        std::vector<std::vector<mlir::Operation*>*> pipelines;
        std::map<mlir::Operation*, size_t> operationToPipelineIx;
        
        /*llvm::outs() << "\n";
        llvm::outs() << "..................................................................";
        llvm::outs() << "\n";
        for (llvm::SmallVector<int8_t> smallVector : edges) {
            llvm::outs() << "(";
            for (int8_t value : smallVector) {
                llvm::outs() << static_cast<int>(value) << ", ";
            }
            llvm::outs() << "), ";
        }
        llvm::outs() << "\n";
        */

        //(0, ), (0, 0, ), (-1, ), (0, 0, ), (-1, ), (-1, ), 

        for (size_t i = 0; i < edges.size(); i++) {
            auto e = edges.at(i);
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

            /*llvm::outs() << "pIx: " << pipelineIx;
            llvm::outs() << "\n";
            v->print(llvm::outs());
            llvm::outs() << "\n";*/

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
            
            //llvm::outs() << "\n";
        }

        /*for (auto pipeline : pipelines) {
            for (auto operation : *pipeline) {
                llvm::outs() << operation->getName().getStringRef().str() << ", ";
            }
            llvm::outs() << "\n";
        }*/
        //llvm::outs() << "\n";
        pipeline_groups.push_back(pipelines);
        pipeline_groups_map.push_back(operationToPipelineIx);
    }

    llvm::outs() << "Pipeline groups size: " << pipeline_groups.size() << "\n";
    
    //check if possible

    std::vector<bool> isValid_structural;
    isValid_structural.reserve(isEdgeActivated_combinations.size());
    
    //check structural validity correctly
    for (size_t i = 0; i < pipeline_groups.size(); i++) {
        auto pipelines = pipeline_groups.at(i);
        auto operationToPipelineIx = pipeline_groups_map.at(i);
        auto edges = isEdgeActivated_combinations.at(i);
        bool valid = true;

        for (size_t j = 0; j < vectOps.size(); j++) {
            auto e = edges.at(j);
            auto v = vectOps.at(j); 
            
            for (size_t k = 0; k < v->getNumOperands(); k++) {
                auto b = e[k];
                auto defOp = v->getOperand(k).getDefiningOp();
                if (b == 1) {
                    if (operationToPipelineIx[v] != operationToPipelineIx[defOp]) {
                        valid = false;
                        break;
                    }
                }
                else if (b == 0) {
                    if (operationToPipelineIx[v] == operationToPipelineIx[defOp]) {
                        valid = false;
                        break;
                    }
                }
            }
            if (valid == false) {
                break;
            }
        }
        isValid_structural.push_back(valid);
    }
    size_t svalid = std::count(isValid_structural.begin(), isValid_structural.end(), true);
    llvm::outs() << "isValid_structural size: " << isValid_structural.size() << ", valid=true: " << svalid << "\n";
    llvm::outs() << "----------------------------------------------------------\n";
    //
#if 0
    for (size_t i = 0; i < pipeline_groups.size(); i++) {
        auto pipelines = pipeline_groups.at(i);
        auto edges = isEdgeActivated_combinations.at(i);
        auto valid = isValid_structural.at(i);

        llvm::outs() << "..................................................................";
        llvm::outs() << "\n";
        for (llvm::SmallVector<int8_t> smallVector : edges) {
            llvm::outs() << "(";
            for (int8_t value : smallVector) {
                llvm::outs() << static_cast<int>(value) << ", ";
            }
            llvm::outs() << "), ";
        }
        llvm::outs() << "\n";

        llvm::outs() << "isValid: " << valid << "\n";
        for (auto pipeline : pipelines) {
            for (auto operation : *pipeline) {
                llvm::outs() << operation->getName().getStringRef().str() << ", ";
            }
            llvm::outs() << "\n";
        } 
        llvm::outs() << "\n";
        llvm::outs() << "\n";
    }
#endif

    std::vector<bool> isValid;
    for (size_t j_i = 0; j_i < pipeline_groups.size(); j_i++) {
        
        if (!isValid_structural.at(j_i)) {
            isValid.insert(isValid.end(), dIxs.size(), false);
            continue;
        }

        auto edges = isEdgeActivated_combinations.at(j_i);

        for (size_t d_i = 0; d_i < dIxs.size(); d_i++) {
            
            auto d = dIxs.at(d_i);

            bool valid = true;
            for (size_t v_i = 0; v_i < vectOps.size(); v_i++) {
                auto v = llvm::dyn_cast<daphne::Vectorizable>(vectOps.at(v_i));
                auto e = edges.at(v_i);

                for (size_t k = 0; k < v->getNumOperands(); k++) {
                    size_t d_v = d.at(v);
                    auto split = v.getVectorSplits()[d_v][k];
                    auto b = e[k];
                    
                    if (b == 1) {
                        auto defOp = llvm::dyn_cast<daphne::Vectorizable>(v->getOperand(k).getDefiningOp());
                        size_t d_defOp = d.at(defOp);
                        auto combine = defOp.getVectorCombines()[d_defOp][0];

                        if (!matchingVectorSplitCombine(split, combine)) {
                            valid = false;
                            break;
                        }
                    }
                }
                if (valid == false) {
                    break;
                }
            }
            isValid.push_back(valid);

        }
    }

    size_t dvalid = std::count(isValid.begin(), isValid.end(), true);
    llvm::outs() << "isValid: " << isValid.size() << ", valid=true: " << dvalid << "\n";
    llvm::outs() << "----------------------------------------------------------\n";

    for (size_t i = 0; i < isValid.size(); i++) { 
    }
    
    return;
}


std::unique_ptr<Pass> daphne::createAllVectorizeComputationsPass() {
    return std::make_unique<AllVectorizeComputationsPass>();
}