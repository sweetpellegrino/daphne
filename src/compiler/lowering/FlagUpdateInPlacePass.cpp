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

#include "ir/daphneir/DaphneUpdateInPlaceAttributes.h"

#include <ir/daphneir/DaphneOps.h.inc>

#include <mlir/Pass/Pass.h>

using namespace mlir;

#include <iostream>



/**
 * @brief XXXXXXXXXXX
 */

bool hasAnyUseAfterCurrentOp(mlir::Operation *op, bool checkLhs = true) {

    //Check if operand is used after the current operation op
    mlir::Value arg = checkLhs ? op->getOperand(0) : op->getOperand(1);
    for (auto *userOp : arg.getUsers()) {
        if (op->isBeforeInBlock(userOp)) {
            return true;
        }
    }

    return false;
 }

 bool isValidMatrixOrFrameType(mlir::Operation *op, bool checkLhs = true) {
    //check if data type is matrix or frame
    //what kind of type is this: !daphne.Matrix<2x2xf64:sp[1.000000e+00]>
    mlir::Value arg = checkLhs ? op->getOperand(0) : op->getOperand(1);
    return (arg.getType().isa<daphne::MatrixType>() || arg.getType().isa<daphne::FrameType>()) && (arg.getType() == op->getResult(0).getType());
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

    // Traverse the operations in the module.
    module.walk([&](Operation *op) {
        
        mlir::daphne::UpdateInPlaceAttrValue operandUpdateInPlace = mlir::daphne::UpdateInPlaceAttrValue::NONE;

        //TODO: change to checking the possiblity of inplace update
        //check if result is matrix or frame? e.g what happens if sqrt of scalar

        //EwUnary and EwBinary
        // if (mlir::daphne::Daphne_EwUnaryOp addOp = dynamic_cast<mlir::AddOp>(op)) {
        //     // The operation is an addition operation
        //     // Handle accordingly
        // }
        // else if (mlir::SubOp subOp = dynamic_cast<mlir::SubOp>(op)) {
        //     // The operation is a subtraction operation
        //     // Handle accordingly
        // }


        //In case of loops: if ssa comes from outside loop, we can not update in place
        //if the ssa comes from inside the loop, we can update in place

        if (op->getName().getStringRef() == "daphne.ewAdd" || op->getName().getStringRef() == "daphne.ewSqrt") {
        //if (op->getName().getStringRef() == "daphne.ewAdd" || op->getName().getStringRef() == "daphne.ewSqrt") {

             //change to internal function with getArg?; for binary getLhs, getRhs

            op->print(llvm::outs());
            llvm::outs() << "\n op name: " << op->getName().getStringRef() << "\n";
            llvm::outs() << "hasOneUse: " << op->hasOneUse() << "\n";
            llvm::outs() << "hasOneUseAfter: " << hasAnyUseAfterCurrentOp(op) << "\n";
            llvm::outs() << "op type: ";
            op->getOperand(0).getType().print(llvm::outs());
            llvm::outs() << "\n result type: "; 
            op->getResult(0).getType().print(llvm::outs());
            llvm::outs() << "\n";

            //Check EwUnary, need to gurantee that atleast one operand exists
            //if (mlir::isa<daphne::EwUnaryOp>(op)) {
            if(op->getName().getStringRef() == "daphne.ewSqrt") {
                
                // Check if the operation has only one use
                //TODO: if the variable is not used ever -> hasOneUse == 0
                //TODO: hasNoUse after
                auto value = op->getOperand(0);

                if (value.hasOneUse() || !hasAnyUseAfterCurrentOp(op)) { //&& value.getUsers().begin() == op) { 
                    if ((value.getType().isa<daphne::MatrixType>() || value.getType().isa<daphne::FrameType>())) {

                        // 
                        //TODO: DataBuffer Dependency
                        //
                        operandUpdateInPlace = mlir::daphne::UpdateInPlaceAttrValue::LHS;
                    }
                }
            }
            //TODO: change to checking if op is EwBinaryOp
            else if (op->getName().getStringRef() == "daphne.ewAdd") {

                bool lhsQualifies = isValidMatrixOrFrameType(op, true)  && !hasAnyUseAfterCurrentOp(op, true);
                bool rhsQualifies = isValidMatrixOrFrameType(op, false) && !hasAnyUseAfterCurrentOp(op, false);

                if (lhsQualifies && rhsQualifies) {
                    operandUpdateInPlace = mlir::daphne::UpdateInPlaceAttrValue::BOTH;
                }
                else if (lhsQualifies) {
                    operandUpdateInPlace = mlir::daphne::UpdateInPlaceAttrValue::LHS;
                }
                else if (rhsQualifies) {
                    operandUpdateInPlace = mlir::daphne::UpdateInPlaceAttrValue::RHS;
                }
               
            }
        }

        if(operandUpdateInPlace != mlir::daphne::UpdateInPlaceAttrValue::NONE) {
            OpBuilder builder(op);
            //op->setAttr(ATTR_UPDATEINPLACE_KEY, builder.getI64IntegerAttr(static_cast<int64_t>(operandUpdateInPlace)));
            op->setAttr(mlir::daphne::UpdateInPlaceAttr::getAttrName(), mlir::daphne::UpdateInPlaceAttr::get(builder.getContext(), operandUpdateInPlace));
        }

    });
    llvm::outs() << "\033[0m";
}

std::unique_ptr<Pass> daphne::createFlagUpdateInPlacePass() {
    return std::make_unique<FlagUpdateInPlacePass>();
}