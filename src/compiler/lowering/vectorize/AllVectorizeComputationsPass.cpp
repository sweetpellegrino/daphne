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
#include "compiler/lowering/vectorize/VectorUtils.h"
#include <stack>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "nlohmannjson/json.hpp"

#include <memory>
#include <utility>
#include <vector>
#include <queue>

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

    void generate_decisionIxs_combinations(std::vector<std::vector<size_t>> &combinations, const std::vector<mlir::Operation *> &ops, std::vector<size_t> _combination, VectorIndex vectIx) {
        if (vectIx == ops.size()) {
            combinations.push_back(_combination);
            return; 
        }

        auto op = llvm::dyn_cast<daphne::Vectorizable>(ops.at(vectIx));
        for (size_t i = 0; i < op.getVectorSplits().size(); i++) {
            _combination.push_back(i);
            generate_decisionIxs_combinations(combinations, ops, _combination, vectIx + 1);
            _combination.pop_back();
        }

    }

    void generate_isEdgeActivated_combinations(std::vector<std::vector<llvm::SmallVector<EdgeStatus>>> &combinations, const std::vector<mlir::Operation *> &ops, std::vector<llvm::SmallVector<EdgeStatus>> _combination, llvm::SmallVector<EdgeStatus> _operands, size_t vectIx, size_t operandIx) {
      
        if (ops.at(vectIx)->getNumOperands() == operandIx) {
            _combination.push_back(_operands);
            _operands = llvm::SmallVector<EdgeStatus>();
            vectIx++;
            operandIx = 0;
        }  

        if (vectIx == ops.size()) {
            combinations.push_back(_combination);
            return; 
        }

        auto defOp = ops.at(vectIx)->getOperand(operandIx).getDefiningOp();
        
        if (std::find(ops.begin(), ops.end(), defOp) == ops.end()) { //block?) {
            _operands.push_back(EdgeStatus::INVALID);
            generate_isEdgeActivated_combinations(combinations, ops, _combination, _operands, vectIx, operandIx + 1);
            _operands.pop_back();
        } else {
            _operands.push_back(EdgeStatus::INACTIVE);
            generate_isEdgeActivated_combinations(combinations, ops, _combination, _operands, vectIx, operandIx + 1);
            _operands.pop_back();

            _operands.push_back(EdgeStatus::ACTIVE);
            generate_isEdgeActivated_combinations(combinations, ops, _combination, _operands, vectIx, operandIx + 1);
            _operands.pop_back();
        }
    }

    void saveToJson(std::vector<bool>& isValid, std::vector<std::vector<size_t>>& dIx_combinations, std::vector<std::vector<llvm::SmallVector<EdgeStatus>>>& isEdgeActivated_combinations) {
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

        //o << std::setw(4) << j << std::endl;
        o << j << std::endl;
    }
}    

void AllVectorizeComputationsPass::runOnOperation()
{

    auto func = getOperation();

    llvm::outs() << "AllVectorizeComputationsPass" << "\n";

    //Step 1: Filter vectorizbale operations
    llvm::outs() << "######## STEP 1 ########" << "\n";

    std::vector<mlir::Operation *> ops;
    func->walk([&](daphne::Vectorizable op) {
        ops.emplace_back(op);
    });
    std::reverse(ops.begin(), ops.end());

    //print vectOps
    for (auto &op : ops) {
        llvm::outs() << "Op: ";
        op->print(llvm::outs());
        llvm::outs() << "\n";
    }

    llvm::outs() << "######## END ########" << "\n";

    //find starting ops
    std::vector<mlir::Operation*> leafOps;
    for (auto op : ops) {
        auto users = op->getUsers();
        bool found = false;
        for (auto u :users) {
            if (std::find(ops.begin(), ops.end(), u) != ops.end()) { 
                found = true;
                break;
            }
        }
        if(!found)
            leafOps.push_back(op);
    }
    
    VectorUtils::DEBUG::drawGraph(leafOps, "graph-max.dot"); 

    //Step 2: Identify merging candidates
    llvm::outs() << "######## STEP 2 ########" << "\n";

    //estimated combinations
    size_t v = ops.size();

    //error in calculation
    //i consider all edges, however the generation of different possiblites is based on the "vectorizable" edges
    size_t e = 0;
    for (auto &op : ops) {
        for (auto t : op->getOperands()) {

            if (llvm::isa<BlockArgument>(t)) 
                continue;            

            if(op->getBlock() != t.getDefiningOp()->getBlock())
                continue;

            if (std::find(ops.begin(), ops.end(), t.getDefiningOp()) != ops.end()) {
                e++;
            }
        }
    }

    int64_t f = (v * std::pow(2, e+1));
    llvm::outs() << "v: " << v << ", " << "e:" << e << "\n";
    llvm::outs() << "Estimated: " << f << "\n";

    std::vector<std::vector<size_t>> dIx_combinations;
    generate_decisionIxs_combinations(dIx_combinations, ops, {}, 0);

    llvm::outs() << "dIx_combinations size: " << dIx_combinations.size() << "\n";

#if 0
    for (auto combination : combinations) {
        for (size_t index : combination) {
            llvm::outs() << index << " ";
        }
        llvm::outs() << "\n";
    }
#endif

    //enum
    //VectorIndex
    std::vector<std::vector<llvm::SmallVector<EdgeStatus>>> isEdgeActivated_combinations;
    std::vector<llvm::SmallVector<EdgeStatus>> _isEdgeActivated;

    generate_isEdgeActivated_combinations(isEdgeActivated_combinations, ops, _isEdgeActivated, {}, 0, 0);

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

        std::map<mlir::Operation*, VectorIndex> decisionIx;
        for(size_t i = 0; i < d.size(); i++) {
            decisionIx.insert({ops.at(i), d.at(i)});
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
            auto v = ops.at(i);

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
                if (e[j] == EdgeStatus::ACTIVE) {
                    auto x = v->getOperand(j).getDefiningOp();
                    //x->print(llvm::outs());
                    //llvm::outs() << "\n";
                    if (operationToPipelineIx.find(x) == operationToPipelineIx.end()) {
                        operationToPipelineIx.insert({x, pipelineIx});
                        pipeline->push_back(x);
                    }
                    else {
                        VectorUtils::mergePipelines(pipelines, operationToPipelineIx, pipelineIx, operationToPipelineIx.at(x));
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

        for (size_t j = 0; j < ops.size(); j++) {
            auto e = edges.at(j);
            auto v = ops.at(j); 
            
            for (size_t k = 0; k < v->getNumOperands(); k++) {
                auto b = e[k];
                auto defOp = v->getOperand(k).getDefiningOp();
                if (b == EdgeStatus::ACTIVE) {
                    if (operationToPipelineIx[v] != operationToPipelineIx[defOp]) {
                        valid = false;
                        break;
                    }
                }
                else if (b == EdgeStatus::INACTIVE) {
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

        if (!VectorUtils::tryTopologicalSortPipelines(pipelines, operationToPipelineIx)) { 
            valid = false;
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
            for (size_t v_i = 0; v_i < ops.size(); v_i++) {
                auto v = llvm::dyn_cast<daphne::Vectorizable>(ops.at(v_i));
                auto e = edges.at(v_i);

                for (size_t k = 0; k < v->getNumOperands(); k++) {
                    size_t d_v = d.at(v);
                    auto split = v.getVectorSplits()[d_v][k];
                    auto b = e[k];
                    
                    if (b == EdgeStatus::ACTIVE) {
                        auto defOp = llvm::dyn_cast<daphne::Vectorizable>(v->getOperand(k).getDefiningOp());
                        size_t d_defOp = d.at(defOp);
                        auto combine = defOp.getVectorCombines()[d_defOp][0];

                        if (!VectorUtils::matchingVectorSplitCombine(split, combine)) {
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

    //provide basic information
    llvm::errs() << "{\"valid\":" << dvalid << "}\n";

    saveToJson(isValid, dIx_combinations, isEdgeActivated_combinations);
    return;
}


std::unique_ptr<Pass> daphne::createAllVectorizeComputationsPass() {
    return std::make_unique<AllVectorizeComputationsPass>();
}