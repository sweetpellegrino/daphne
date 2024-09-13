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
    // Class functions
    //-----------------------------------------------------------------

    struct OneVectorizeComputationsPass : public PassWrapper<OneVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
        const DaphneUserConfig& userConfig;

        explicit OneVectorizeComputationsPass(const DaphneUserConfig& cfg) : userConfig(cfg) {}
    };

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
                    VectorUtils::mergePipelines(pipelines, operationToPipelineIx, pipelineIx, operationToPipelineIx.at(x));
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
    VectorUtils::DEBUG::printPipelines(vectOps, operationToPipelineIx, decisionIx, filename);

    VectorUtils::createVectorizedPipelineOps(func, _pipelines, decisionIx);
 
    return;
}


std::unique_ptr<Pass> daphne::createOneVectorizeComputationsPass(const DaphneUserConfig& cfg) {
    return std::make_unique<OneVectorizeComputationsPass>(cfg);
}