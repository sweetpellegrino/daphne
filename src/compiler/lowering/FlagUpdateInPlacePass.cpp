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
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include "compiler/utils/CompilerUtils.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Pass/Pass.h>

#include "ir/daphneir/DaphneUpdateInPlaceOpInterface.h"

using namespace mlir;

#include <iostream>



/**
 * @brief XXXXXXXXXXX
 */

bool hasAnyUseAfterCurrentOp(mlir::Operation *op, int operand_index) {

    //Check if operand is used after the current operation op
    mlir::Value arg = op->getOperand(operand_index);
    for (auto *userOp : arg.getUsers()) {
        if (op->isBeforeInBlock(userOp)) {
            return true;
        }
    }

    return false;
 }

template<typename T>
 bool isValidType(T arg) {
     return arg.getType().template isa<daphne::MatrixType>() || arg.getType().template isa<daphne::FrameType>();
 }

struct FlagUpdateInPlacePass: public PassWrapper<FlagUpdateInPlacePass, OperationPass<ModuleOp>>
{
    //explicit FlagInPlace() {}
    void runOnOperation() final;
};

void FlagUpdateInPlacePass::runOnOperation() {

    auto module = getOperation();

    llvm::outs() << "\033[1;31m";
    llvm::outs() << "FlagUpdateInPlacePass\n";

    // Traverse the operations in the module, if InPlaceable.
    module.walk([&](mlir::Operation *op) {

        if (auto inPlaceOp = llvm::dyn_cast<daphne::InPlaceable>(op)) {
        
            auto inPlaceOperands = inPlaceOp.getInPlaceOperands();
            BoolAttr inPlaceFutureUse[inPlaceOperands.size()];

            for (auto inPlaceOperand : inPlaceOperands) {
                //TODO: Checking if the operand is valid type (matrix & frame) really necessary? We need to do it also in RewriteToCallKernelOpPass.cpp
                if (!isValidType(op->getOperand(inPlaceOperand)) || hasAnyUseAfterCurrentOp(op, inPlaceOperand))
                    inPlaceFutureUse[inPlaceOperand] = BoolAttr::get(op->getContext(), true);
                else
                    inPlaceFutureUse[inPlaceOperand] = BoolAttr::get(op->getContext(), false);
            }

            //add inPlaceFutureUse to the op as an attribute
            llvm::MutableArrayRef<mlir::Attribute> inPlaceFutureUseArray(inPlaceFutureUse, inPlaceOperands.size());
            op->setAttr("inPlaceFutureUse", mlir::ArrayAttr::get(op->getContext(), inPlaceFutureUseArray));

        }

    });
    llvm::outs() << "\033[0m";
}

std::unique_ptr<Pass> daphne::createFlagUpdateInPlacePass() {
    return std::make_unique<FlagUpdateInPlacePass>();
}