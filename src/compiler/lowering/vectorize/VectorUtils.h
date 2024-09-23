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

#pragma once
#include "compiler/lowering/vectorize/simdjson.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include <cstddef>
#include <iostream>
#include <fstream>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <stack>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>
#include <algorithm>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ManagedStatic.h"

using VectorIndex = std::size_t;
using Pipeline = std::vector<mlir::Operation*>; 
using PipelinePair = std::pair<Pipeline*, Pipeline*>;

namespace std {
    template <>
    struct hash<PipelinePair> {
        size_t operator()(const PipelinePair& p) const {
            return std::hash<Pipeline*>{}(p.first) ^ std::hash<Pipeline*>{}(p.second);
        }
    };
}

enum class DisconnectReason {
    NONE,
    MULTIPLE_CONSUMERS,
    INVALID
};

enum class EdgeStatus {
    INVALID,
    ACTIVE,
    INACTIVE
};

/*
class VectorIndex {
private:
    size_t vectorIndex = 0;
    bool isUnknown = true;
public:
    VectorIndex() : vectorIndex(0), isUnknown(true) {}
    VectorIndex(size_t Ix) : vectorIndex(Ix), isUnknown(false) {}

    size_t get() {
        return vectorIndex;
    }

};*/

#if 0
struct VectorizedOperation {
    VectorizedOperation(mlir::Operation* op, VectorIndex vectIx) : op(op), vectIx(vectIx) {}

    mlir::Operation* op;
    VectorIndex vectIx;

    bool operator==(const VectorizedOperation& other) const {
        return (this->op == other.op) && (this->vectIx == other.vectIx);
    }

};

struct FusionCandidate {
    enum class Type {
        HORIZONTAL,
        PRODUCER_CONSUMER
    };

    FusionCandidate(VectorizedOperation op1, VectorizedOperation op2, Type type) : 
        op1(op1), op2(op2), type(type) {}

    VectorizedOperation op1;
    VectorizedOperation op2;

    Type type;

    //change to method
    [[maybe_unused]] friend bool operator==(const FusionCandidate& c1, const FusionCandidate& c2) {
        if (c1.type != c2.type)
            throw std::runtime_error("Comparison of fusion candidates with different types is not allowed.");

        if (c1.type == FusionCandidate::Type::HORIZONTAL) {
            return (c1.op1 == c2.op1 && c1.op2 == c2.op2) ||
                    (c1.op1 == c2.op2 && c1.op2 == c2.op1);
        } else {
            return (c1.op1 == c2.op1 && c1.op2 == c2.op2);
        }
    }
};

namespace std {
    template <>
    struct hash<VectorizedOperation> {
        size_t operator()(const VectorizedOperation& v) const noexcept {
            return (hash<mlir::Operation *>()(v.op) << 1) + hash<VectorIndex>()(v.vectIx);
        }
    };

    template <>
    struct hash<FusionCandidate> {
        size_t operator()(const FusionCandidate& c) const noexcept {
            if (c.type == FusionCandidate::Type::HORIZONTAL) {
                return hash<VectorizedOperation>()(c.op1) ^ hash<VectorizedOperation>()(c.op2);
            } else {
                return (hash<VectorizedOperation>()(c.op1) << 1) + hash<VectorizedOperation>()(c.op1) + hash<VectorizedOperation>()(c.op2);
            }
        }
    };
}
#endif

/*class Pipeline {
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

    friend Pipeline* mergePipelines(std::vector<Pipeline*>& pipelines, std::map<mlir::Operation*, Pipeline*>& operationToPipeline, Pipeline* pipe1, Pipeline* pipe2){
        //llvm::outs() << mergeFromIx << " " << mergeIntoIx << "\n";
        if (pipe1 == pipe2)
            return nullptr;

        auto pipe1Ops = &pipe2->operations;
        for (auto op : pipe2->operations) {
            //this check is necessary in case we wanted to allow for duplicate calculations
            if  (std::find(pipe1Ops->begin(), pipe1Ops->end(), op) == pipe1Ops->end()) {
                pipe1Ops->push_back(op);
                operationToPipeline[op] = pipe1;
            }

        }
        //merge decision maps
        pipe1->decisions.insert(pipe2->decisions.begin(), pipe2->decisions.end());

        //remove 
        auto pipeIx2 = std::find(pipelines.begin(), pipelines.end(), pipe2);
        pipelines.erase(pipeIx2);
        return pipe1;
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

};*/


struct FusionPair {
    enum class Type {
        HORIZONTAL,
        PRODUCER_CONSUMER
    };

    FusionPair(mlir::Operation *op1, mlir::Operation *op2, Type type) : op1(op1), op2(op2), type(type) {}

    mlir::Operation *op1;
    mlir::Operation *op2;
    Type type;

    [[maybe_unused]] friend bool operator==(const FusionPair& c1, const FusionPair& c2) {
        if (c1.type != c2.type)
            throw std::runtime_error("Comparison of fusion candidates with different types is not allowed.");

        if (c1.type == FusionPair::Type::HORIZONTAL) {
            return (c1.op1 == c2.op1 && c1.op2 == c2.op2) ||
                    (c1.op1 == c2.op2 && c1.op2 == c2.op1);
        } else {
            return (c1.op1 == c2.op1 && c1.op2 == c2.op2);
        }
    }

    void print() const {
        std::string type_str;
        if (type == FusionPair::Type::HORIZONTAL) {
            type_str = "Horizontal";
        } else {
            type_str = "Producer-Consumer";
        }

        std::string op1_str = "nullptr";
        if (op1 != nullptr)
            op1_str = op1->getName().getStringRef().str();

        std::string op2_str = "nullptr";
        if (op2 != nullptr)
            op2_str = op2->getName().getStringRef().str();

        llvm::outs() << "FusionPair: Type=" << type_str;
        llvm::outs() << ", Op1=" << op1_str;
        llvm::outs() << ", Op2=" << op2_str << "\n";
    }
};

namespace std {
    template <>
    struct hash<FusionPair> {
        size_t operator()(const FusionPair& c) const noexcept {
            if (c.type == FusionPair::Type::HORIZONTAL) {
                return hash<mlir::Operation *>()(c.op1) ^ hash<mlir::Operation *>()(c.op2);
            } else {
                return (hash<mlir::Operation *>()(c.op1) << 1) + hash<mlir::Operation *>()(c.op1) + hash<mlir::Operation *>()(c.op2);
            }
        }
    };
}

struct VectorUtils {
    
    static bool matchingVectorSplitCombine(mlir::daphne::VectorSplit split, mlir::daphne::VectorCombine combine) {
        //lvm::outs() << split << " " << combine << "\n";
        mlir::daphne::VectorCombine _operandCombine;
        switch (split) {
            case mlir::daphne::VectorSplit::ROWS:
                _operandCombine = mlir::daphne::VectorCombine::ROWS;
                break;
            case mlir::daphne::VectorSplit::COLS:
                _operandCombine = mlir::daphne::VectorCombine::COLS;
                break;
            default:
                // No matching split/combine; basically resulting in separate pipelines
                return false;
        }
        if (combine == _operandCombine)
            return true;
        return false;
    }

    //------------------------------------------------------------------------------

    static std::pair<std::vector<mlir::Operation*>, std::vector<mlir::Operation*>> getConnectedOpsBetweenPipelines(Pipeline* pipe1, Pipeline* pipe2) {

        std::vector<mlir::Operation*> _io;
        std::vector<mlir::Operation*> _oo;

        for (auto op : *pipe1) {
            bool isInput = true;
            bool isOutput = true;

            for (const auto operand : op->getOperands()) {
                mlir::Operation* defOp = operand.getDefiningOp(); 

                //check if feasible for blockargs
                if (!defOp)
                    continue;
                else {
                    if (std::find(pipe2->begin(), pipe2->end(), defOp) == pipe2->end()) {
                        isInput = false;
                        break;
                    }
                }
            }

            for (const auto user : op->getUsers()) {
                if (std::find(pipe2->begin(), pipe2->end(), user) == pipe2->end()) {
                    isOutput = false;
                    break;
                }
            }

            if (isInput)
                _io.push_back(op);
            if (isOutput)
                _oo.push_back(op);
        }
        return {_io, _oo};
    }

    //Function merges two pipelines into one by appending all operations from one pipeline to another
    //Order is not really considered, as it is embodied in IR
    static void mergePipelines(std::vector<Pipeline*>& pipelines, std::map<mlir::Operation*, size_t>& operationToPipelineIx, size_t pipeIx1, size_t pipeIx2){
        //llvm::outs() << mergeFromIx << " " << mergeIntoIx << "\n";
        if (pipeIx1 == pipeIx2)
            return;
        if (pipeIx2 > pipeIx1)
            std::swap(pipeIx1, pipeIx2);

        std::vector<mlir::Operation*> *mergedPipeline(pipelines.at(pipeIx2));
        for (auto op : *pipelines.at(pipeIx1)) {
            if  (std::find(mergedPipeline->begin(), mergedPipeline->end(), op) == mergedPipeline->end()) {
                mergedPipeline->push_back(op);
                operationToPipelineIx[op] = pipeIx2;
            }
        }
        pipelines.at(pipeIx2) = std::move(mergedPipeline);
        pipelines.erase(pipelines.begin() + pipeIx1);
    }

    static Pipeline* mergePipelines(std::vector<Pipeline*>& pipelines, std::map<mlir::Operation*, Pipeline*>& operationToPipeline, Pipeline* pipe1, Pipeline* pipe2){
        if (pipe1 == pipe2)
            return nullptr;

        for (auto op : *pipe2) {
            if  (std::find(pipe1->begin(), pipe1->end(), op) == pipe1->end()) {
                pipe1->push_back(op);
                operationToPipeline[op] = pipe1;
            }

        }

        auto pipeIx2 = std::find(pipelines.begin(), pipelines.end(), pipe2);
        pipelines.erase(pipeIx2);
        return pipe1;
    }

    //Function merges two pipelines into one by appending all operations from one pipeline to another
    //Order is not really considered, as it is embodied in IR
    static size_t mergePipelines(std::vector<std::vector<mlir::Operation*>>& pipelines, std::map<mlir::Operation*, size_t>& operationToPipelineIx, size_t pipeIx1, size_t pipeIx2){
        //llvm::outs() << mergeFromIx << " " << mergeIntoIx << "\n";
        /*if (pipeIx1 == pipeIx2)
            return pipeIx1;

        if (pipeIx2 > pipeIx1)
            std::swap(pipeIx1, pipeIx2);*/

        std::vector<mlir::Operation*> mergedPipeline(pipelines.at(pipeIx2));
        for (auto op : pipelines.at(pipeIx1)) {
            if  (std::find(mergedPipeline.begin(), mergedPipeline.end(), op) == mergedPipeline.end()) {
                mergedPipeline.push_back(op);
                operationToPipelineIx[op] = pipeIx2;
            }
        }
        pipelines.at(pipeIx2) = std::move(mergedPipeline);
        pipelines.erase(pipelines.begin() + pipeIx1);

        //returns the Ix the pipeline that does not longer exist was merged into
        return pipeIx2;
    }

    //------------------------------------------------------------------------------

    static bool arePipelinesConnected(const std::vector<std::vector<mlir::Operation*>>& pipelines, const std::map<mlir::Operation*, size_t>& operationToPipelineIx, size_t pipeIx1, size_t pipeIx2){

        //false better?
        if (pipeIx1 == pipeIx2)
            return true;
        
        auto pipe1 = pipelines[pipeIx1];
        for (auto op : pipe1) {
            for (auto operandValue : op->getOperands()) {
                auto defOp = operandValue.getDefiningOp();
                if (operationToPipelineIx.find(defOp) != operationToPipelineIx.end()) {
                    auto defOpPipeIx = operationToPipelineIx.at(defOp);
                    if (defOpPipeIx == pipeIx2) {
                        return true;
                    }
                }
            } 
        }

        //other way, as potentially bidirectional relationship
        //we are checking only the operands not results
        auto pipe2 = pipelines[pipeIx2];
        for (auto op : pipe2) {
            for (auto operandValue : op->getOperands()) {
                auto defOp = operandValue.getDefiningOp();
                if (operationToPipelineIx.find(defOp) != operationToPipelineIx.end()) {
                    auto defOpPipeIx = operationToPipelineIx.at(defOp);
                    if (defOpPipeIx == pipeIx1) {
                        return true;
                    }
                }
            } 
        }

        return false;
    }       

    //TODO: improve
    static bool arePipelinesConnected(const std::map<PipelinePair, DisconnectReason> pcr, Pipeline* pipe1, Pipeline* pipe2){

        if (pipe1 == pipe2)
            return true;
        
        std::stack<Pipeline*> s; 
        std::vector<Pipeline*> visited;

        s.push(pipe1);
        while (!s.empty()) {
            Pipeline* currPipe = s.top(); s.pop();

            //Connection found
            if (currPipe == pipe2) 
                return true;

            for (const auto& pipePair : pcr) {
                if (pipePair.first.first == currPipe && std::find(visited.begin(), visited.end(), pipePair.first.first) == visited.end()) {
                    s.push(pipePair.first.second); 
                }
            }
            visited.push_back(currPipe);
        }

        return false;
    }       

    static bool tryTopologicalSortMerged(std::vector<Pipeline*> &pipelines, std::map<PipelinePair, DisconnectReason> &rel, Pipeline* pipe1, Pipeline* pipe2) {

        //if (pipe2 > pipe1)
        //  std::swap(pipe1, pipe2);

        //prealloc
        std::map<Pipeline*, std::unordered_set<Pipeline*>> pipeline_graph;
        for (auto pipe : pipelines) {
            if (pipe == pipe1)
                pipe = pipe2; 
            pipeline_graph.insert({pipe ,{}});
        }

        for (auto& [key, _] : rel) {
            auto consumer = key.second;    
            auto producer = key.first;    

            if (consumer == pipe1) {
                consumer = pipe2;
            }
            else if (producer == pipe1) {
                producer = pipe2;
            }

            if (producer == consumer)
                continue;

            if (pipeline_graph.find(consumer) == pipeline_graph.end()) {
                pipeline_graph.insert({consumer, {producer}});
            } else {
                pipeline_graph.at(consumer).insert(producer);
            }
        }

        /*for (auto node : pipeline_graph) {
            llvm::outs() << "Key: " << node.first << ", Values: ";
            for (auto dependency : node.second) {
                llvm::outs() << dependency << " ";
            }
            llvm::outs() << "\n";
        }
        llvm::outs() << "\n";*/

        return tryTopologicalSort(pipeline_graph);

    }

    static std::map<PipelinePair, DisconnectReason> consolidateProducerConsumerRelationship(std::multimap<PipelinePair, DisconnectReason> mmPCR) {
        std::map<PipelinePair, DisconnectReason> pcr;
        for (const auto& [pipePair, disReason] : mmPCR) {
            if (pcr.find(pipePair) == pcr.end()) {
                pcr.insert({pipePair, disReason});
            }
            else {
                //Overwrite if INVALID as it domiantes MULTI_CONSUMER relationship
                if(disReason == DisconnectReason::INVALID) {
                    pcr.insert_or_assign(pipePair, disReason);
                }
            }
        }
        return pcr;
    }

    static bool tryTopologicalSortPreMergedPipelines(const std::vector<std::vector<mlir::Operation*>>& pipelines, const std::map<mlir::Operation*, size_t>& operationToPipelineIx, size_t pipeIx1, size_t pipeIx2) {

        if (pipeIx2 > pipeIx1)
            std::swap(pipeIx1, pipeIx2);

        std::map<size_t, std::unordered_set<size_t>> pipeline_graph;

        for (size_t i = 0; i < pipelines.size(); i++) {

            auto pipeline = pipelines.at(i);

            std::unordered_set<size_t> incomingPipelinesIxs;
            for (auto op : pipeline) {
                size_t currIx = operationToPipelineIx.at(op);

                if (currIx != i)
                    throw std::runtime_error("integrity error");

                if (currIx == pipeIx2)
                    currIx = pipeIx1;

                for (auto operand : op->getOperands()) {
                    auto defOp = operand.getDefiningOp();
                    if(operationToPipelineIx.find(defOp) == operationToPipelineIx.end())
                        continue;

                    auto defOpIx = operationToPipelineIx.at(defOp);

                    if (defOpIx == pipeIx1)
                        defOpIx = pipeIx2;
                    
                    if(defOpIx != currIx) {
                        op->print(llvm::outs());
                        llvm::outs() << "\n";
                        defOp->print(llvm::outs());
                        llvm::outs() << "\n";
                        incomingPipelinesIxs.insert(defOpIx);
                    }
                }
            }
            pipeline_graph.insert({i, incomingPipelinesIxs});
        }

        for (auto node : pipeline_graph) {
            llvm::outs() << "Key: " << node.first << ", Values: ";
            for (auto dependency : node.second) {
                llvm::outs() << dependency << " ";
            }
            llvm::outs() << "\n";
        }
        llvm::outs() << "\n";

        return tryTopologicalSort(pipeline_graph);

    }

    static bool tryTopologicalSortPipelines(const std::vector<std::vector<mlir::Operation*>*>& pipelines, const std::map<mlir::Operation*, size_t> operationToPipelineIx) {

        std::map<size_t, std::unordered_set<size_t>> pipeline_graph;

        for (size_t i = 0; i < pipelines.size(); i++) {
            auto pipeline = pipelines.at(i);

            std::unordered_set<size_t> incomingPipelinesIxs;
            for (auto op : *pipeline) {
                size_t currIx = operationToPipelineIx.at(op);
                if (currIx != i) {
                    throw std::runtime_error("integrity error");
                }
                for (auto operand : op->getOperands()) {
                    auto defOp = operand.getDefiningOp();
                    if(operationToPipelineIx.find(defOp) == operationToPipelineIx.end())
                        continue;
                    auto defOpIx = operationToPipelineIx.at(defOp);
                    if(defOpIx != currIx) {
                        incomingPipelinesIxs.insert(defOpIx);
                    }
                }
            }
            pipeline_graph.insert({i, incomingPipelinesIxs});
        }

        return tryTopologicalSort(pipeline_graph);
         
    }

    private:
    //kahn: https://dev.to/leopfeiffer/topological-sort-with-kahns-algorithm-3dl1
    //https://leetcode.com/problems/course-schedule/solutions/483330/c-kahns-algorithm-topological-sort-with-easy-detailed-explanation-16-ms-beats-98/
    static bool tryTopologicalSort(std::map<size_t, std::unordered_set<size_t>> pipeline_graph) {

        std::unordered_map<size_t, size_t> inDegrees;
        for (auto node : pipeline_graph) {
            for (auto dependency : node.second) {
                ++inDegrees[dependency];
            }
        }

        std::queue<size_t> queue;
        for (auto node : pipeline_graph) {
            if (inDegrees[node.first] == 0) {
                queue.push(node.first);
            }
        }

        std::vector<size_t> result;
        while (!queue.empty()) {
            size_t node = queue.front(); queue.pop();
            result.push_back(node);
            for (auto dependency : pipeline_graph.at(node)) {
                if (--inDegrees[dependency] == 0) {
                    queue.push(dependency);
                }
            }
        }

        return result.size() == pipeline_graph.size();
    }

    static bool tryTopologicalSort(std::map<Pipeline*, std::unordered_set<Pipeline*>> pipeline_graph) {

        std::unordered_map<Pipeline*, size_t> inDegrees;
        for (auto node : pipeline_graph) {
            for (auto dependency : node.second) {
                ++inDegrees[dependency];
            }
        }

        std::queue<Pipeline*> queue;
        for (auto node : pipeline_graph) {
            if (inDegrees[node.first] == 0) {
                queue.push(node.first);
            }
        }

        std::vector<Pipeline*> result;
        while (!queue.empty()) {
            Pipeline* node = queue.front(); queue.pop();
            result.push_back(node);
            for (auto dependency : pipeline_graph.at(node)) {
                if (--inDegrees[dependency] == 0) {
                    queue.push(dependency);
                }
            }
        }

        return result.size() == pipeline_graph.size();
    }

    public:
    /**
     * @brief Recursive function checking if the given value is transitively dependant on the operation `op`.
     * @param value The value to check
     * @param op The operation to check
     * @return true if there is a dependency, false otherwise
     */
    static bool valueDependsOnResultOf(mlir::Value value, mlir::Operation *op) {
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
    static void movePipelineInterleavedOperations(mlir::Block::iterator pipelinePosition, const std::vector<mlir::Operation*> pipeline) {
        // first operation in pipeline vector is last in IR, and the last is the first
        auto startPos = pipeline.back()->getIterator();
        auto endPos = pipeline.front()->getIterator();
        auto currSkip = pipeline.rbegin();

        std::vector<mlir::Operation*> moveBeforeOps;
        std::vector<mlir::Operation*> moveAfterOps;

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

    static void createVectorizedPipelineOps(mlir::func::FuncOp func, std::vector<Pipeline> pipelines, std::map<mlir::Operation*, VectorIndex> decisionIxs) {
        mlir::OpBuilder builder(func);


        // Create the `VectorizedPipelineOp`s
        for(auto _pipeline : pipelines) {
            if(_pipeline.empty())
                continue;
            
            auto valueIsPartOfPipeline = [&](mlir::Value operand) {
                return llvm::any_of(_pipeline, [&](mlir::Operation* lv) { return lv == operand.getDefiningOp(); });
            };
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
            VectorUtils::movePipelineInterleavedOperations(builder.getInsertionPoint(), _pipeline);

            //potential addition for
            std::vector<mlir::Operation*> pipeline;
            for(auto vIt = _pipeline.rbegin(); vIt != _pipeline.rend(); ++vIt) {
                auto v = *vIt;

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
                        vSplitAttrs.push_back(mlir::daphne::VectorSplitAttr::get(func.getContext(), vSplits[i]));
                        operands.push_back(operand);
                    }
                }

                // Determination of results of VectorizedPipelineOps!
                for(auto vCombine : vCombines) {
                    vCombineAttrs.push_back(mlir::daphne::VectorCombineAttr::get(func.getContext(), vCombine));
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
                        auto m1x1 = mlir::daphne::MatrixType::get(func.getContext(), r.getType(), 1, 1, 1, mlir::daphne::MatrixRepresentation::Dense);
                        auto loc = v->getLoc();

                        auto toCastOp = builder.create<mlir::daphne::CastOp>(loc, m1x1, r);
                        toCastOp->moveAfter(v);
                        
                        //xxxxxx
                        pipeline.push_back(toCastOp);
                        vCombineAttrs.push_back(mlir::daphne::VectorCombineAttr::get(func.getContext(), vCombines[i]));
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
            /*if (!mlir::sortTopologically(bodyBlock)) {
                throw std::runtime_error("topoSort");
            } */  
        }
    }
    
    //-----------------------------------------------------------------
    // 
    //-----------------------------------------------------------------

    struct DEBUG {

        static void drawGraph(const std::vector<mlir::Operation*> leafOps, std::string filename) {
            std::stack<mlir::Operation*> stack;
            std::ofstream dot(filename);
            if (!dot.is_open()) {
                throw std::runtime_error("test");
            }

            dot << "digraph G {\n";
            for (auto s : leafOps) {
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

                auto v = llvm::dyn_cast<mlir::daphne::Vectorizable>(op);
                if (!v) 
                    continue;

                for (size_t i = 0; i < v->getNumOperands(); ++i) {
                    mlir::Value e = v->getOperand(i);

                    if (llvm::isa<mlir::BlockArgument>(e)) 
                        continue;

                    auto defOp = e.getDefiningOp();
                    
                    if (llvm::isa<mlir::daphne::MatrixType>(e.getType()) && llvm::isa<mlir::daphne::Vectorizable>(defOp)) {
                        dot << "\"" << defOp->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(defOp) << "\"";
                        dot << " -> ";
                        dot << "\"" << op->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(op) << "\" [label=\"" << i << "\"];\n";
                        stack.push(defOp);
                    }
                }
            }
            dot << "}";
            dot.close();
        }

        static std::string getColor(size_t pipelineId) {
            std::vector<std::string> colors = {"tomato", "lightgreen", "lightblue", "plum1", "mistyrose2", "seashell", "hotpink",
                                            "lemonchiffon", "firebrick1", "ivory2", "khaki1", "lightcyan", "olive", "yellow",
                                            "maroon", "violet", "navajowhite1"};
            return colors[pipelineId % colors.size()];
        }

        static void drawPipelines(const std::vector<mlir::Operation*> &ops, const std::map<mlir::Operation*, size_t> &operationToPipelineIx, const std::map<mlir::Operation*, VectorIndex> &decisionIxs, std::string filename) {
            std::ofstream outfile(filename);

            outfile << "digraph G {" << std::endl;

            std::map<mlir::Operation*, std::string> opToNodeName;

            for (size_t i = 0; i < ops.size(); ++i) {
                std::string nodeName = "node" + std::to_string(i);
                opToNodeName[ops.at(i)] = nodeName;

                size_t pipelineId = operationToPipelineIx.at(ops[i]);
                VectorIndex vectIx = decisionIxs.at(ops.at(i));
                std::string color = VectorUtils::DEBUG::getColor(pipelineId);

                outfile << nodeName << " [label=\"" << ops.at(i)->getName().getStringRef().str() << "\\npIx: " << pipelineId << ", vectIx: " << vectIx << "\", fillcolor=" << color << ", style=filled];" << std::endl;
            }

            std::unordered_set<mlir::Operation*> outsideOps; 

            for (size_t i = 0; i < ops.size(); ++i) {
                mlir::Operation* op = ops.at(i);
                auto consumerPipelineIx = operationToPipelineIx.at(op);

                for (const auto& operandValue : op->getOperands()) {
                    mlir::Operation* operandOp = operandValue.getDefiningOp();
                    auto it = operationToPipelineIx.find(operandOp);

                    if (it != operationToPipelineIx.end()) {
                        auto producerPipeplineIx = it->second;
                        outfile << opToNodeName.at(operandOp) << " -> " << opToNodeName.at(op);

                        if (producerPipeplineIx != consumerPipelineIx) {
                            outfile << " [style=dotted]";
                        }
                        outfile << ";" << std::endl;
                    }
                    else {
                        //also show the surrounding ops, e.g. to make horizontal fusion visible
                    } 
                }
            }
            outfile << "}" << std::endl;
        }

        static std::string printPtr(void *ptr) {

            std::ostringstream oss;
            oss << std::hex << reinterpret_cast<uintptr_t>(ptr);
            
            std::string str = oss.str();
            
            return str.substr(str.size() - 3);
        }


        static void drawPipelines(const std::vector<mlir::Operation*> &ops, const std::map<mlir::Operation*, Pipeline*> &operationToPipeline, const std::map<mlir::Operation*, VectorIndex> &decisionIxs, std::string filename) {
            std::ofstream outfile(filename);

            outfile << "digraph G {" << std::endl;

            std::map<mlir::Operation*, std::string> opToNodeName;
            std::map<Pipeline*, size_t> pipelineToIx;

            for (size_t i = 0; i < ops.size(); ++i) {
                std::string nodeName = "node" + std::to_string(i);
                opToNodeName[ops.at(i)] = nodeName;

                auto pipeline = operationToPipeline.at(ops.at(i));
                size_t pipelineIx;
                if (pipelineToIx.find(pipeline) == pipelineToIx.end()) {
                    pipelineIx = pipelineToIx.size();
                    pipelineToIx.insert({pipeline, pipelineIx});
                }
                else {
                    pipelineIx = pipelineToIx.at(pipeline);
                }
                std::string color = VectorUtils::DEBUG::getColor(pipelineIx);
                VectorIndex vectIx = decisionIxs.at(ops.at(i));

                std::string pipeName = printPtr(pipeline);

                outfile << nodeName << " [label=\"" << ops.at(i)->getName().getStringRef().str() << "\\npIx: " << pipeName << ", vectIx: " << vectIx << "\", fillcolor=" << color << ", style=filled];" << std::endl;
            }

            std::unordered_set<mlir::Operation*> outsideOps; 

            for (size_t i = 0; i < ops.size(); ++i) {
                mlir::Operation* op = ops.at(i);
                auto consumerPipelineIx = operationToPipeline.at(op);

                for (const auto& operandValue : op->getOperands()) {
                    mlir::Operation* operandOp = operandValue.getDefiningOp();
                    auto it = operationToPipeline.find(operandOp);

                    if (it != operationToPipeline.end()) {
                        auto producerPipeplineIx = it->second;
                        outfile << opToNodeName.at(operandOp) << " -> " << opToNodeName.at(op);

                        if (producerPipeplineIx != consumerPipelineIx) {
                            outfile << " [style=dotted]";
                        }
                        outfile << ";" << std::endl;
                    }
                    else {
                        //also show the surrounding ops, e.g. to make horizontal fusion visible
                    } 
                }
            }
            outfile << "}" << std::endl;
        }

        static void printPCR(std::map<PipelinePair,  DisconnectReason> pcr) {
            for (const auto& [key, value]  : pcr) {
                llvm::outs() << "(" << VectorUtils::DEBUG::printPtr(key.first) << ", " << VectorUtils::DEBUG::printPtr(key.second) << "|" << int(value) << ")\n";
            } 
            llvm::outs() << "\n";
        }

        static void printPipelines (std::vector<Pipeline*> pipelines) {
            for (auto p : pipelines) {
                llvm::outs() << VectorUtils::DEBUG::printPtr(p) << " ";
                for (auto op : *p) {
                    llvm::outs() << op->getName().getStringRef().str() << ", ";
                }
                llvm::outs() << "\n";
            }
        }

    };


    struct BENCH {

    };
};