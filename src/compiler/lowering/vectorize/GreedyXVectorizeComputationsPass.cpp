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
#include <cstddef>
#include <map>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <util/ErrorHandler.h>
#include "compiler/lowering/vectorize/VectorUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Passes.h"
#include <stack>
#include <queue>

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace
{

    //-----------------------------------------------------------------
    // Helper Classes
    //-----------------------------------------------------------------
    
    class HCandidate {
    public:
        mlir::Operation *op1;
        size_t decisionIx_op1;
        mlir::Operation *op2; 
        size_t decisionIx_op2;
        HCandidate(mlir::Operation *op1, size_t decisionIx_op1, mlir::Operation *op2, size_t decisionIx_op2) 
            : op1(op1), decisionIx_op1(decisionIx_op1), op2(op2), decisionIx_op2(decisionIx_op2) {}

        [[maybe_unused]] friend bool operator==(const HCandidate& c1, const HCandidate& c2) {
            return (c1.op1 == c2.op1 && c1.decisionIx_op1 == c2.decisionIx_op1 &&
                    c1.op2 == c2.op2 && c1.decisionIx_op2 == c2.decisionIx_op2) ||
                    (c1.op1 == c2.op2 && c1.decisionIx_op1 == c2.decisionIx_op2 &&
                    c1.op2 == c2.op1 && c1.decisionIx_op2 == c2.decisionIx_op1);
        }

        void print() const {
            llvm::outs() << "HCandidate:\n";
            llvm::outs() << "op1: " << op1->getName().getStringRef().str() << ", decisionIx_op1: " << decisionIx_op1 << "\n";
            llvm::outs() << "op2: " << op2->getName().getStringRef().str() << ", decisionIx_op2: " << decisionIx_op2 << "\n";
        }

    };

    // {ewMul, <add, sqrt>, 0, <0, 0>}
    class PCCandidate {
    public:
        mlir::Operation *op; //consumer
        std::vector<mlir::Operation*> operandOps; //producers

        size_t decisionIx_op;
        std::vector<size_t> decisionIx_operandOps;

        PCCandidate(mlir::Operation* op, std::vector<mlir::Operation*> operandOps, size_t decisionIx_op, std::vector<size_t> decisionIx_operandOps)
            : op(op), operandOps(operandOps), decisionIx_op(decisionIx_op), decisionIx_operandOps(decisionIx_operandOps) {}

        void print() const {
            llvm::outs() << "PCCandidate:" << "\n";
            llvm::outs() << "op: " << op->getName().getStringRef().str() << "\n";
            llvm::outs() << "operandOps: ";
            for (auto* op : operandOps) {
                if (op != nullptr) {
                    llvm::outs() << op->getName().getStringRef().str() << " ";
                } else {
                    //nullptr means the operandOp is not a valid vectorizable producer (cf. constrains)
                    llvm::outs() << "OUTSIDE" << " ";
                }
            }
            llvm::outs() << "\ndecisionIx_op: " << decisionIx_op << "\n";
            llvm::outs() << "decisionIx_operandOps: ";
            for (auto ix : decisionIx_operandOps) {
                llvm::outs() << ix << " ";
            }
            llvm::outs() << "\n";
        }
    };

    class OpDec {
    public:
        mlir::Operation* op;
        size_t decisionIx;
        OpDec(mlir::Operation* op, size_t decisionIx)
            : op(op), decisionIx(decisionIx) {}
       
        bool operator==(const OpDec& other) const {
            return (this->op == other.op) && (this->decisionIx == other.decisionIx);
        }

        void print() const {
            if (op != nullptr)
                llvm::outs() << "(" << op->getName().getStringRef().str() << ", " << decisionIx << ")";
            else
                llvm::outs() << "(" << "NULLPTR" << ", " << decisionIx << ")";
        } 
    };
}

namespace std {
    template<>
    struct hash<HCandidate> {
        std::size_t operator()(const HCandidate& c) const {
            //https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
            //careful if c.op1 == c.op2 defaults to 0, for all op
            return (hash<mlir::Operation*>()(c.op1) ^ hash<size_t>()(c.decisionIx_op1)) ^
                   (hash<mlir::Operation*>()(c.op2) ^ hash<size_t>()(c.decisionIx_op2));
        }
    };
    template<> 
    struct hash<OpDec> {
        size_t operator()(const OpDec& opDec) const {
            return ((hash<mlir::Operation*>()(opDec.op) ^ (hash<size_t>()(opDec.decisionIx) << 1)) >> 1);
        }
    };
}


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

    struct GreedyXVectorizeComputationsPass : public PassWrapper<GreedyXVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;

        //Function that modifies existing mlir programm structure
        //Currently only gets a pipeline as a list of nodes | cf. Formalisation
        void createVectorizedPipelineOps(func::FuncOp func, std::vector<std::vector<mlir::Operation *>> pipelines, std::map<mlir::Operation*, size_t> decisionIxs) {
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
                for(auto vIt = pipeline.rbegin(); vIt != pipeline.rend(); ++vIt) {
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
                    //TODO
                    if (auto allAggSumOp = llvm::dyn_cast<daphne::AllAggSumOp>(v)) { 
                        auto m1 = daphne::MatrixType::get(allAggSumOp.getContext(), allAggSumOp->getResult(0).getType(), 1, 1, 1, daphne::MatrixRepresentation::Dense);
                        m1.print(llvm::outs());
                        llvm::outs() << "\n";
                        allAggSumOp->getResult(0).setType(m1); 

                        //create castOp for casting the 1x1 matrix to scalar value
                        auto loc = allAggSumOp->getLoc();
                        auto castOp = builder.create<daphne::CastOp>(loc, m1.getElementType(), allAggSumOp->getResult(0));
                        allAggSumOp->getResult(0).replaceAllUsesExcept(castOp.getResult(), castOp);

                        //movePipelineInterleavedOperations is before of exisiting of this op
                        castOp->moveAfter(allAggSumOp);
                        castOp.dump();
                    }

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
            }
        }
    };

    //Function merges two pipelines into one by appending all operations from one pipeline to another
    //Order is not really considered, as it is embodied in IR
    void mergePipelines(std::vector<std::vector<mlir::Operation*>>& pipelines, std::map<mlir::Operation*, size_t>& operationToPipelineIx, size_t mergeFromIx, size_t mergeIntoIx){
        llvm::outs() << mergeFromIx << " " << mergeIntoIx << "\n";
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

    void mergePipelines(std::vector<mlir::Operation*>& pipeline1, std::vector<mlir::Operation*>& pipeline2, std::map<mlir::Operation*, size_t>& operationToPipelineIx) {

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

    void printGraph(std::unordered_map<OpDec, std::vector<OpDec>> primaryCandidates, std::string filename) {
        std::ofstream dot(filename);
        if (!dot.is_open()) {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        dot << "digraph G {\n";

        for (auto [_key, _value] : primaryCandidates) {
            auto op = _key.op;
            auto d = _key.decisionIx;

            if (op != nullptr) {
                // Print the node
                dot << "\"" << op->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(op) << ":" << d << "\" [label=\"" << op->getName().getStringRef().str() << "\\n" <<d << "\"];\n";

                for (auto od : _value) {
                    if(od.op == nullptr)
                        dot << "\"" << "nullptr" << "+" << std::hex << reinterpret_cast<uintptr_t>(od.op) << ":" << od.decisionIx << "\" -> \"" << op->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(op) << ":" << d << "\";\n";
                    else
                        dot << "\"" << op->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(op) << ":" << d << "\" -> \"" << od.op->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(od.op) << ":" << od.decisionIx << "\";\n";
                }
            } else {
                llvm::errs() << "Warning: Null pointer operation encountered.\n";
            }
        }

        dot << "}\n";
        dot.close();
    }

    void backward_propagation(std::vector<std::vector<mlir::Operation*>> *pipelines, std::map<mlir::Operation*, size_t> *decisionIxs) {

        std::stack<std::pair<mlir::Operation*, size_t>> stack;
        std::unordered_set<mlir::Operation*> visited;
        auto _op = pipelines->at(0).at(0);

        stack.push({_op, 0});

        while (!stack.empty()) {
            auto t = stack.top(); stack.pop();
            mlir::Operation* op = t.first;
            size_t d = t.second;

            if(std::find(visited.begin(), visited.end(), op) != visited.end())
                continue;

            auto v = llvm::dyn_cast<daphne::Vectorizable>(op);

            visited.insert(op);
            decisionIxs->insert({op, d});

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
    }

    void expandCombinations(std::vector<std::vector<size_t>>& operandsCombineDecisionIxs, std::vector<std::vector<size_t>>& decisionCombinations, std::vector<size_t> &_combination, size_t operandIx) {
        if (operandIx == operandsCombineDecisionIxs.size()) {
            decisionCombinations.push_back(_combination);
            return;
        }

        auto operandDecisionIxs =  operandsCombineDecisionIxs[operandIx];
        for (size_t i = 0; i < operandDecisionIxs.size(); i++) {
            _combination.push_back(operandDecisionIxs[i]);
            expandCombinations(operandsCombineDecisionIxs, decisionCombinations, _combination, operandIx+1);
            _combination.pop_back();
        }
    }

    bool doesOpSecExistsAsProducer(const OpDec& od, const std::unordered_map<OpDec, std::vector<OpDec>>& primaryCandidates) {
        for (auto [key, value] : primaryCandidates) {
            if (std::any_of(value.begin(), value.end(),
                            [&od](const OpDec& val) { return val == od; })) {
                return true;
            }
        }
        return false;
    }

}
    
void GreedyXVectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    llvm::outs() << "GreedyXVectorizeComputationsPass" << "\n";

    //Step 1: Filter vectorizbale operations
    //Slice Analysis for reverse topological sorting?
    //Good illustration here: PostOrder https://mlir.llvm.org/doxygen/SliceAnalysis_8h_source.html
    //Implementation of walk here: https://mlir.llvm.org/doxygen/Visitors_8h_source.html#l00062

    llvm::outs() << "######## STEP 1 ########" << "\n";

    std::vector<daphne::Vectorizable> vectOps;
    func->walk([&](daphne::Vectorizable op) {
        vectOps.emplace_back(op);
    });
    std::reverse(vectOps.begin(), vectOps.end());

    //print vectOps
    for (auto op : vectOps) {
        llvm::outs() << "Op: ";
        op->print(llvm::outs());
        llvm::outs() << "\n";
    }

    llvm::outs() << "######## END ########" << "\n";

    //Improvment: can we already know that some operations can not be fused???
    //Step 2: Identify merging candidates
    llvm::outs() << "######## STEP 2 ########" << "\n";

    //TODO: use llvm::SmallVector? is optimized small arrays
    //(daphne.transpose, 1), <(NULLPTR, 0), >
    //(daphne.transpose, 0), <(NULLPTR, 0), >
    //(daphne.ewAdd, 0), <(daphne.ewSqrt, 0), (daphne.transpose, 1), >
    //(daphne.ewSqrt, 1), <(NULLPTR, 0), >
    //(daphne.sumAll, 1), <(daphne.ewAdd, 1), >
    //(daphne.ewSqrt, 0), <(NULLPTR, 0), >
    //(daphne.ewAdd, 1), <(daphne.ewSqrt, 1), (daphne.transpose, 0), >
    //(daphne.sumAll, 0), <(daphne.ewAdd, 0), >
    std::unordered_map<OpDec, std::vector<OpDec>> primaryCandidates;
    std::vector<OpDec> insertOrder; //not needed? as we could derive the necessary information from vectOps?
    //TODO: Should this make a weak connection? So in case of not being greedy; first to broken up, if necessary
    //when combining the individual steps together to make the algorithm more efficient these candidates,
    //could still be a separate step, as it potentially inhibits the heursitic to find an optimal pipeline 
    //(think about the split points in case of collision for layout/access propagation)
    std::unordered_set<HCandidate> secondaryCandidates;

    //identify the last op
    std::unordered_map<mlir::Operation*, bool> isLast;
    isLast.reserve(vectOps.size());

    //Assumption:
    //That what we can Vectorize, we want to Vectorize
    //A pipeline should be able to be executed / must be valid
    //reversed vectOps
    for (auto opv : vectOps) {

        if (isLast.find(opv) == isLast.end()) {
            isLast[opv] = true;
        }

        //Starting point are the different possibilites to choose from for a Vectorizable op
        for(size_t decisionIx = 0; decisionIx < opv.getVectorSplits().size(); decisionIx++) {
            auto splits = opv.getVectorSplits()[decisionIx];
            
            /*for (auto s : splits) {
                llvm::outs() << s << " "; 
            }
            llvm::outs() << "\n";*/

            //opv->getOperands() implicitly provides the provider consumer relationship
            auto operandSplitPairs = llvm::zip(opv->getOperands(), splits);
            //needs to be vector of vector if we have multiple possiblities for e.g. row combine
            //e.g. matmul with row combine over lhs row, rhs br OR lhs br, rhs row split
            //change to daphne::Vectorizable?
            std::vector<std::vector<size_t>> operandsCombineDecisionIxs;
            std::vector<mlir::Operation*> operandDefOps;
            //operandCombineDecisionIx.reserve(opv->getNumOperands());
            for (auto [operand, split] : operandSplitPairs) {
                //Here okay, as we don't "broadcast" smaller vectors by expanding them
                //Also excludes scalars, careful if considering source ops (like fill), this early exit will probably not work anymore
                if (split == daphne::VectorSplit::NONE)
                    continue;

                auto defOp = operand.getDefiningOp();
                auto v_defOp = llvm::dyn_cast<daphne::Vectorizable>(defOp);

                //-----------------------------------------------------------------
                // Producer -> Consumer
                //-----------------------------------------------------------------

                //only considering producer / consumer relationship if both are vectorizable
                //if not these are canidates at the top of pipeline and does contain a producer to consider
                if (v_defOp && v_defOp->getBlock() == opv->getBlock()) {
                    //follow compability for producer / consumer relationship
                    for (size_t operandDecisionIx = 0; operandDecisionIx < v_defOp.getVectorCombines().size(); operandDecisionIx++) {
                        //currently only considering one return cf. [0]
                        auto operandCombine  = v_defOp.getVectorCombines()[operandDecisionIx][0];
                        if (VectorUtils::matchingVectorSplitCombine(split, operandCombine)) {
                            auto it = std::find(operandDefOps.begin(), operandDefOps.end(), defOp);
                            isLast[defOp] = false;
                            if (it == operandDefOps.end()) {
                                operandDefOps.push_back(defOp);
                                operandsCombineDecisionIxs.push_back({operandDecisionIx});
                            } else {
                                auto ix = std::distance(operandDefOps.begin(), it);
                                operandsCombineDecisionIxs[ix].push_back(operandDecisionIx);
                            }
                        }
                        else {
                            llvm::outs() << "HIT" << "\n";
                        }
                        //for completion here also nullptr writeback?
                        //then i also need it for VectorSplit::NONE
                        //does not work trivially
                        /*else {
                            operandDefOps.push_back(nullptr);
                            operandsCombineDecisionIxs.push_back({0});
                        }*/
                    }
                } else {
                    //storing nullptr for hinting the pipeline builder
                    //currently a nullptr only exist if it completly outside
                    operandDefOps.push_back(nullptr);
                    operandsCombineDecisionIxs.push_back({0});
                }

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
                for (auto user : operand.getUsers()) {
                    
                    if (user == opv)
                        continue;

                    auto v_user = llvm::dyn_cast<daphne::Vectorizable>(user);
                    if (!v_user)
                        continue; 

                    //We need to check if opv and user are not in a producer / consumer relationship
                    //Can we use the information from above to skip?
                    bool is_only_horizontal = true;
                    for (auto rel : user->getOperands()) {
                        if (rel.getDefiningOp() == opv) {
                            is_only_horizontal = false;
                            break;
                        }
                    }

                    if (!is_only_horizontal)
                        continue;

                    size_t userOperandIx = 0;
                    for (auto use : user->getOperands()) {
                        if (use == operand) {
                            break;
                        }
                        userOperandIx++;
                    }

                    //the (other) user and the current opv should have the same split
                    //as if not, it basically results in a broadcast (the other needs col the other one needs row) <- is this true?
                    for (size_t userDecisionIx = 0; userDecisionIx < v_user.getVectorSplits().size(); userDecisionIx++) {
                        //currently only considering one return cf. [0]
                        auto userSplit = v_user.getVectorSplits()[userDecisionIx][userOperandIx];
                        if (split == userSplit) {
                            secondaryCandidates.insert({opv, decisionIx, user, userDecisionIx});
                        }
                    } 
                }
                
            }

            std::vector<std::vector<size_t>> decisionCombinations; 
            std::vector<size_t> _combination;

            expandCombinations(operandsCombineDecisionIxs, decisionCombinations, _combination, 0);

            for (size_t i = 0; i < decisionCombinations.size(); i++) {
                if (!decisionCombinations[i].empty()) {
                    OpDec od = {opv, decisionIx};
                    if (primaryCandidates.find(od) != primaryCandidates.end()) {
                        //probably need to change to std::vector<std::vector<OpDec>> as value
                        throw std::runtime_error("error"); 
                    }
                    std::vector<OpDec> opDecs;
                    for (size_t j = 0; j < decisionCombinations[i].size(); j++) {
                        opDecs.push_back({operandDefOps[j], decisionCombinations[i][j]});
                    }
                    primaryCandidates.insert({od, opDecs});
                    insertOrder.push_back(od);
                }
            }
        }
    }
    llvm::outs() << "######## END ########" << "\n";

    for (auto [_key, _value] : primaryCandidates) {
        _key.print();
        llvm::outs() << ", <"; 
        for (auto od : _value) {
            od.print();
            llvm::outs() << ", "; 
        }
        llvm::outs() << ">\n"; 
    }

    for (auto x : secondaryCandidates) {
        x.print();
        llvm::outs() << "\n";
    }

    for (auto kv : isLast) {
        std::cout << "op: " << kv.first->getName().getStringRef().str() << ", is?: " << kv.second << "\n";
    }

    //Step 3: Exploratory merge pipelines
    llvm::outs() << "######## STEP 3 ########" << "\n";

    //get all latest ops

#if 1

    //(daphne.transpose, 1), <(NULLPTR, 0), >
    //(daphne.transpose, 0), <(NULLPTR, 0), >
    //(daphne.ewAdd, 0), <(daphne.ewSqrt, 0), (daphne.transpose, 1), >
    //(daphne.ewSqrt, 1), <(NULLPTR, 0), >
    //(daphne.sumAll, 1), <(daphne.ewAdd, 1), >
    //(daphne.ewSqrt, 0), <(NULLPTR, 0), >
    //(daphne.ewAdd, 1), <(daphne.ewSqrt, 1), (daphne.transpose, 0), >
    //(daphne.sumAll, 0), <(daphne.ewAdd, 0), >

    std::vector<OpDec> leafOpDecs;
    std::vector<mlir::Operation*> leafOps;

    for (const auto [key, _] : primaryCandidates) {
        if (!doesOpSecExistsAsProducer(key, primaryCandidates)) {
            leafOpDecs.push_back(key); 
            leafOps.push_back(key.op); 
        }
    }
    
    printGraph(leafOps, "graph.dot");
    printGraph(primaryCandidates, "graph_pc.dot");

    std::vector<std::vector<std::vector<mlir::Operation*>>> pipelines_groups;
    std::vector<std::map<mlir::Operation*, size_t>> decisionIxs;

    std::queue<OpDec> q;

    //pipeline group == single solution
    return;
    for (auto od : leafOpDecs) {
        llvm::outs() << "New Pipeline Group\n";
        std::unordered_set<mlir::Operation*> visited;
        std::vector<std::vector<mlir::Operation*>> pipelines;
        std::vector<mlir::Operation*> _pipeline;
        std::map<mlir::Operation*, size_t> _decisionIxs;


        llvm::outs() << "Starting OpDec: ";
        od.print();
        llvm::outs() << "\n";

        //introduction of a pipelinestate
        q.push(od); //(daphne.sumAll, 0)

        bool finished = false;
        llvm::outs() << "\nwhile finished\n";
        while (!finished) {
            llvm::outs() << "while empty\n";
            while (!q.empty()) {
                auto od = q.front();

                llvm::outs() << "Visiting OpDec: ";
                od.print();

                //check if we already have inserted the op in a pipeline
                if(std::find(visited.begin(), visited.end(), od.op) != visited.end()) {
                    llvm::outs() << "\n";
                    llvm::outs() << "op already visted\n";
                    // here we need to check if matching
                    //if pipeline includes the "OpDec" than merge into existing Pipeline
                    //
                    q.pop();
                    continue;
                }

                visited.insert(od.op);
                _pipeline.push_back(od.op);

                //get the prevalidated producers for the current OpDec
                if (primaryCandidates.find(od) != primaryCandidates.end()) {

                    auto producersOfOpDec = primaryCandidates.at(od);

                    llvm::outs() << ", fetched: "; 
                    od.print();
                    llvm::outs() << ", <"; 
                    for (auto od : producersOfOpDec) {
                        od.print();
                        llvm::outs() << ", "; 
                    }
                    llvm::outs() << ">\n"; 

                    //primaryCandidates.at(od) == <(daphne.ewAdd, 0), >
                    for(auto prod_od : producersOfOpDec) {
                        //this can happen if od describes a top node
                        if (prod_od.op == nullptr)
                            continue; 
                        q.push(prod_od);
                    }
                } else {
                    llvm::outs() << "??" << "\n";
                }
                q.pop();
            }

            //check if all vectOps where visted
            std::unordered_set<mlir::Operation*> missing;
            for (auto vecOp : vectOps) {
                if (visited.find(vecOp.getOperation()) == visited.end()) {
                    missing.insert(vecOp.getOperation());
                }
            }
            for (auto mis : missing) {
                for (auto [_key, _] : primaryCandidates) {
                    if(mis == _key.op) {
                        //q();
                    }
                }
            }
            if (missing.empty()) {
                finished = true;
                llvm::outs() << "finished" << "\n";
            }
            finished = false;
        }

        llvm::outs() << "\n";
        llvm::outs() << "\n";

    }
    //bascially after finishing following the graph, we need to identify vectOps that was not visited

#endif

    llvm::outs() << "######## END ########" << "\n";

    //printGraph(pipelines[0][0], "test.dot");

    //Step 5: Small pipeline merge, if possible? why not further try to reduce the number of individuals pipelines 
    // and their overhead (e.g. runtime) and merge together if constraints are met (input dimension)
    //Till now we didn´t need to check if dimension matches as they do by definition of considered operators and checked relationship
    //Full potential, if we allow for different output types?

    //careful as it takes the assumption that the size is equal for every object
    //in case of "broadcasting" we need to make it different
    //TODO: check what about SourceOps
#if 0
    llvm::outs() << "######## STEP 5 ########" << "\n";
    auto lambda = [](std::vector<mlir::Operation*> pipeline){
        for (auto op : pipeline) {
            for (auto operandType : op->getOperandTypes()) {
                operandType.print(llvm::outs());
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

    //unfathomable dirty
    for (size_t i = 0; i < pipelines.size(); ++i) {
        for (size_t j = i + 1; j < pipelines.size(); ++j) {
            if (lambda(pipelines[i]) == lambda(pipelines[j])) {
                mergePipelines(pipelines, operationToPipelineIx, j, i);
            }
        }
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

    //Step X-2: Data access propagation
    //Start at a dominant operator like matmul (latest in execution first?)
    //do backwards propagation: from operator to input for determination of vector splits of the inputs
    //do forwads propagation: from operator to output for determination of vector combine of the output
    //when there is a collision, we need to backtrack, if backtrack fails we definitly need to split the pipeline
    //(is there an earlier way to identify necessary splits? / after n backtracks force split?)
    //where do we need to split? where first collision happens? 
    //how do we want to propagate?: dfs, bfs -> do we really need the std::vector<Pipelines>? (could combine with earlier steps)
    //is here the mlir dataflow framework applicable?


    //What happens if we don't have any dominant operators or
    //due to horizontal fusion: we could have disconnected graph inside the pipeline
    //where do we start then? at the end? how to efficiently identify the end?

    /*for(auto op : dominant_operations_in_pipeline) {
        //std::map<mlir::Operation*, bool>* visited = new std::map<mlir::Operation*, bool>();
        //std::vector<mlir::Operation*> *visited = new std::vector<mlir::Operation*>();
        for (auto d_item : backward_propagation(op)) {
            llvm::outs() << "op: " << d_item.first->getName().getStringRef() << ", decision: " << d_item.second << "\n";
        }
        
        forward_propagation();
    }
    */

    auto decisionIxs = std::map<mlir::Operation*, size_t>();
    backward_propagation(&pipelines, &decisionIxs);
    for (auto d_item : decisionIxs) {
        llvm::outs() << "op: " << d_item.first->getName().getStringRef() << ", decision: " << d_item.second << "\n";
    }

    //Step 4: Horizontal Fusion
    //Separate step as it allows for the producer -> consumer relationship to be exploited first
    //Where does it make a difference?
    // What about size information and broadcast of the sharing operator: does it make sense if matrix too small? all inputs need to be
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

    //Step X-1: Data layout propagation?
    //it is probably better to switch the order with data access later on as we allow for an optimnization for individual kernels
    // first before employing the access, which is in most cases a means to an end? atleast for elementwise operators, check for operators working on a specific axis
    //combine it into one step with data access propagation?

    //Step X: create pipeline ops
    GreedyXVectorizeComputationsPass::createVectorizedPipelineOps(func, pipelines, decisionIxs);
#endif
}


std::unique_ptr<Pass> daphne::createGreedyXVectorizeComputationsPass() {
    return std::make_unique<GreedyXVectorizeComputationsPass>();
}