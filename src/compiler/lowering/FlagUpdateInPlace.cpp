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
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <ir/daphneir/DaphneOps.h.inc>

#include <mlir/Pass/Pass.h>

using namespace mlir;

#include <iostream>

const std::string ATTR_UPDATEINPLACE = "updateInPlace";

/**
 * @brief XXXXXXXXXXX
 */

bool hasOneUseAfter(mlir::Operation *op) {

    //Check if operand is used after the operation
    auto value = op->getOperand(0);
    for (Operation *userOp : value.getUsers()) {
        if (userOp != op) {
            return true;
        }
    }

    return false;
 }

struct FlagUpdateInPlace: public PassWrapper<FlagUpdateInPlace, OperationPass<ModuleOp>>
{
    //explicit FlagInPlace() {}
    void runOnOperation() final;
};

void FlagUpdateInPlace::runOnOperation() {

    auto module = getOperation();

    llvm::outs() << "\033[1;31m";

    // Traverse the operations in the module.
    module.walk([&](Operation *op) {
        
        bool qualifiesForUpdateInPlace = false;

        //TODO: change to checking the possiblity of inplace update
        //check if result is matrix or frame?

        //EwUnary and EwBinary

        if (op->hasTrait<OpTrait::UpdateInPlace>()) {
        //if (op->getName().getStringRef() == "daphne.ewAdd" || op->getName().getStringRef() == "daphne.ewSqrt") {

            op->print(llvm::outs());
            llvm::outs() << "\n op name: " << op->getName().getStringRef() << "\n";
            llvm::outs() << "hasOneUse: " << op->hasOneUse() << "\n";

            //Check EwUnary, need to gurantee that atleast one operand exists
            //if (mlir::isa<daphne::EwUnaryOp>(op)) {
            if(op->getName().getStringRef() == "daphne.ewSqrt") {
                
                // Check if the operation has only one use
                //TODO: if the variable is not used ever -> hasOneUse == 0
                //TODO: hasNoUse after

                auto value = op->getOperand(0);
                if (value.hasOneUse() || hasOneUseAfter(op)) { //&& value.getUsers().begin() == op) { 
                    // Check if the input data object is not used in any future operations
                    // Value input = op->getOperand(0);
                    // input.print(llvm::outs());
                    // llvm::outs() << " value \n";
                    // for (Operation *userOp : input.getUsers()) {
                    // llvm::outs() << "userOp:" << userOp  << "\n";
                    // }
                    
                    // 
                    //TODO: DataBuffer Dependency
                    //

                    qualifiesForUpdateInPlace = true;
                }
            }

            // if (mlir::daphne::Daphne_EwUnaryOp addOp = dynamic_cast<mlir::AddOp>(op)) {
            //     // The operation is an addition operation
            //     // Handle accordingly
            // }
            // else if (mlir::SubOp subOp = dynamic_cast<mlir::SubOp>(op)) {
            //     // The operation is a subtraction operation
            //     // Handle accordingly
            // }
            
        }
    
        //Flag the operation indicating that update-in-place optimization can be applied.
        if (qualifiesForUpdateInPlace) {
            OpBuilder builder(op);
            op->setAttr(ATTR_UPDATEINPLACE, builder.getBoolAttr(true));
            //change operand to a wrapper type?
        }

    });
    llvm::outs() << "\033[0m";
}

std::unique_ptr<Pass> daphne::createFlagUpdateInPlace() {
    return std::make_unique<FlagUpdateInPlace>();
}