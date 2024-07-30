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

#ifndef SRC_COMPILER_LOWERING_VECTORIZE_VECTORIZECOMPUTATIONSUTILS_H
#define SRC_COMPILER_LOWERING_VECTORIZE_VECTORIZECOMPUTATIONSUTILS_H
#include <util/ErrorHandler.h>

using namespace mlir;

enum VectorizeType {
    DAPHNE,
    GREEDY,
    TH_GREEDY
};

/**
    * @brief Recursive function checking if the given value is transitively dependant on the operation `op`.
    * @param value The value to check
    * @param op The operation to check
    * @return true if there is a dependency, false otherwise
    */
inline bool valueDependsOnResultOf(Value value, Operation *op) {
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
inline void movePipelineInterleavedOperations(Block::iterator pipelinePosition, const std::vector<mlir::Operation*> &pipeline) {
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

#endif //SRC_COMPILER_LOWERING_VECTORIZE_VECTORIZECOMPUTATIONSUTILS_H

