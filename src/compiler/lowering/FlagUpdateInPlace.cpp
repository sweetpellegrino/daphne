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

#include "compiler/lowering/AttributeDefinitions.h"

#include <ir/daphneir/DaphneOps.h.inc>

#include <mlir/Pass/Pass.h>

using namespace mlir;

#include <iostream>



/**
 * @brief XXXXXXXXXXX
 */

/*
bool hasAnyUseBeforeDBDep(mlir::Operation *op, bool checkLhs = true) {

    //Check if operand is used after the current operation op
    mlir::Value arg = checkLhs ? op->getOperand(0) : op->getOperand(1);
    for (auto *userOp : arg.getUsers()) {
        if (userOp->hasTrait<OpTrait::VirtualResult>()
            && userOp->isBeforeInBlock(op) ) {
            return true;
        }
    }

    return false;
 }
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
        
        ATTR_UPDATEINPLACE_TYPE operandUpdateInPlace = ATTR_UPDATEINPLACE_TYPE::NONE;

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

        if (op->hasTrait<OpTrait::UIPUnary>() || op->hasTrait<OpTrait::UIPBinary>()) {
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
            if(op->hasTrait<OpTrait::UIPUnary>()) {
                
                // Check if the operation has only one use
                //TODO: if the variable is not used ever -> hasOneUse == 0
                //TODO: hasNoUse after
                auto value = op->getOperand(0);

                if (value.hasOneUse() || !hasAnyUseAfterCurrentOp(op)) { //&& value.getUsers().begin() == op) { 
                    if ((value.getType().isa<daphne::MatrixType>() || value.getType().isa<daphne::FrameType>())) {

                        // 
                        //TODO: DataBuffer Dependency
                        //
                        operandUpdateInPlace = ATTR_UPDATEINPLACE_TYPE::LHS;
                    }
                }
            }
            //TODO: change to checking if op is EwBinaryOp
            else if (op->hasTrait<OpTrait::UIPBinary>()) {

                bool lhsQualifies = isValidMatrixOrFrameType(op, true)  && !hasAnyUseAfterCurrentOp(op, true);
                bool rhsQualifies = isValidMatrixOrFrameType(op, false) && !hasAnyUseAfterCurrentOp(op, false);

                if (lhsQualifies && rhsQualifies) {
                    operandUpdateInPlace = ATTR_UPDATEINPLACE_TYPE::BOTH;
                }
                else if (lhsQualifies) {
                    operandUpdateInPlace = ATTR_UPDATEINPLACE_TYPE::LHS;
                }
                else if (rhsQualifies) {
                    operandUpdateInPlace = ATTR_UPDATEINPLACE_TYPE::RHS;
                }
               
            }
        }
    
        if(operandUpdateInPlace != ATTR_UPDATEINPLACE_TYPE::NONE) {
            OpBuilder builder(op);
            op->setAttr(ATTR_UPDATEINPLACE_KEY, builder.getI64IntegerAttr(static_cast<int64_t>(operandUpdateInPlace)));
        }

    });
    llvm::outs() << "\033[0m";
}

std::unique_ptr<Pass> daphne::createFlagUpdateInPlace() {
    return std::make_unique<FlagUpdateInPlace>();
}