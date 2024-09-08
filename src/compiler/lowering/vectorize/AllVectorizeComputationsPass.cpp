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
#include "nlohmannjson/json.hpp"

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
    // Class functions
    //-----------------------------------------------------------------

    struct AllVectorizeComputationsPass : public PassWrapper<AllVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
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

    void saveToJson(std::vector<bool>& isValid, std::vector<std::vector<size_t>>& dIx_combinations, std::vector<std::vector<llvm::SmallVector<int8_t>>>& isEdgeActivated_combinations) {
        nlohmann::json j;

        int count = 0;
        for(size_t i = 0; i < isValid.size(); ++i) {
            if (isValid.at(i)) {
                std::string key = std::to_string(count); 
                int k = i / dIx_combinations.size();
                int l = i % dIx_combinations.size(); 
                j[key]["dIx"] = nlohmann::json(dIx_combinations[l]);
                j[key]["isEdgeActive"] = nlohmann::json(isEdgeActivated_combinations[k]);
                count++;
            }
        }

        std::ofstream o("data.json");

        o << std::setw(4) << j << std::endl;
    }

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

    saveToJson(isValid, dIx_combinations, isEdgeActivated_combinations);
    return;
}


std::unique_ptr<Pass> daphne::createAllVectorizeComputationsPass() {
    return std::make_unique<AllVectorizeComputationsPass>();
}