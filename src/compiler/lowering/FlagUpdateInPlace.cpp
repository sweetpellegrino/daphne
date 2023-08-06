/*
 * Copyright 2023 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/IR/User.h"
#include "llvm/Support/raw_ostream.h"
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/Pass/Pass.h>

using namespace mlir;

#include <iostream>

/**
 * @brief Inserts profiling tracepoints
 */
struct FlagUpdateInPlace: public PassWrapper<FlagUpdateInPlace, OperationPass<ModuleOp>>
{
    //explicit FlagInPlace() {}
    void runOnOperation() final;
};

void FlagUpdateInPlace::runOnOperation()
{

    auto module = getOperation();

    //walk over all operations, check wether allows for update-in-place and check the operand dependency
    module->walk([&](Operation *op) {
        llvm::outs() << "\033[1;31m";

        op->print(llvm::outs());
        llvm::outs() << "\n";

        llvm::outs() << "Found a flagged operation: " << op->getName().getStringRef() << "\n";
        llvm::outs() << "Number of operands: " << op->getNumOperands() << "\n";
        llvm::outs() << "Number of results: " << op->getNumResults() << "\n";
        llvm::outs() << "Number of regions: " << op->getNumRegions() << "\n";
        llvm::outs() << "Number of successors: " << op->getNumSuccessors() << "\n";

        llvm::outs() << "\033[0m";

        for (int i = 0; i < op->getNumOperands(); ++i) {
            auto operand = op->getOperand(i);
            llvm::outs() << "Operand: " << operand << "\n";
            
            for (auto *user : operand.getUsers()) {
                llvm::outs() << "User: " << user << "\n";
                llvm::outs() << "User name: " << user->getName().getStringRef() << "\n";
                llvm::outs() << "User has " << user->getNumOperands() << " operands\n";

                for (int j = 0; j < user->getNumOperands(); ++j) {
                    auto userOperand = user->getOperand(j);
                    llvm::outs() << "User operand: " << userOperand << "\n";
                    if (userOperand == operand) {
                        llvm::outs() << "Found a dependency: " << operand << " -> " << user << "\n";
                    }
                }
            }
        }

    });
    
}

std::unique_ptr<Pass> daphne::createFlagUpdateInPlace() {
    return std::make_unique<FlagUpdateInPlace>();
}