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


#include "compiler/utils/CompilerUtils.h"
#include <util/ErrorHandler.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <set>
#include <iostream>

#include <iostream>
#include <vector>

#include "compiler/lowering/vectorize/VectorizeComputationsBase.h"

using namespace mlir;

namespace
{
    bool isVectorizable(func::FuncOp &op) {
        return true;
    }

    //fetch matching combinations

    bool isFusible(daphne::Vectorizable opi, daphne::Vectorizable opv) {
        if (opi->getBlock() != opv->getBlock())
            return false;


        return true;
    }
    
    struct GreedyVectorizeComputationsPass : public PassWrapper<GreedyVectorizeComputationsPass, OperationPass<func::FuncOp>> {
        void runOnOperation() final;
    };

    //Split: before producer
    //Combine: after consumer
    struct VectorOption {
        daphne::VectorSplit split;
        daphne::VectorCombine combine;
    };

    struct VectorCandidate {
        daphne::Vectorizable producer;
        daphne::Vectorizable consumer;
        std::vector<VectorOption> options;
    };
}

void GreedyVectorizeComputationsPass::runOnOperation()
{
    std::cout << "GREEDY" << std::endl;
    auto func = getOperation();

    //Step 1: Reverse Topological Sorting & Filtering of Vectorizable functions
    //TODO: Vereine (reverse) topological sort mit dem Filtern
    std::vector<daphne::Vectorizable> vectOps;
    func->walk<WalkOrder::PostOrder>([&](mlir::Operation *op) {
        llvm::outs() << "func name: " << op->getName() << "\n";
        if(auto vec = llvm::dyn_cast<daphne::Vectorizable>(op)) {
            llvm::outs() << "vec func name: " << op->getName() << "\n";
        }
    });

    //Step 2: Identify merging candidates
    std::vector<VectorCandidate> candidates;
    for (auto opv : vectOps) {
        //Get incoming edges of operation opv
        auto args = opv->getOperands();

        //For every operation of incoming argument of opv, check if isFusible
        //True: push into possible merge candidates list
        for(auto arg : args) {
            
            //Check if isFusible
            if(auto opi = arg.getDefiningOp<daphne::Vectorizable>()) {
                if (isFusible(opi, opv)) {
                    VectorCandidate candidate {opi, opv};
                    candidates.push_back(candidate);
                }
            }
        }
    }

    //Step 3: Greedy merge pipelines 


    //Step X: create pipeline ops

}

std::unique_ptr<Pass> daphne::createGreedyVectorizeComputationsPass() {
    return std::make_unique<GreedyVectorizeComputationsPass>();
}