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

    // TODO: fuse pipelines that have the matching inputs, even if no output of the one pipeline is used by the other.
    //  This requires multi-returns in way more cases, which is not implemented yet.

    // Find vectorizable operations and their inputs of vectorizable operations
    std::vector<daphne::Vectorizable> vectOps;
    module->walk([&](daphne::Vectorizable op)
    {
        llvm::outs() << "Vectorizable\n";
      if(CompilerUtils::isMatrixComputation(op))
          vectOps.emplace_back(op);
    });
    std::vector<daphne::Vectorizable> vectorizables(vectOps.begin(), vectOps.end());
    for(auto v : vectorizables) {
     v.getVectorSplits();
    } 

    std::vector<daphne::InPlaceable> vecs;
     module->walk([&](daphne::InPlaceable op)
    {
       vecs.emplace_back(op);
    });

    for (auto v : vecs) {
        v.GetInPlaceOperands();
    }

    // Traverse the operations in the module.
    module.walk([&](mlir::Operation *op) {
        
        mlir::daphne::InPlaceEnum inPlaceAttr = mlir::daphne::InPlaceEnum::NONE;

        //TODO: change to checking the possiblity of inplace update
        //check if result is matrix or frame? e.g what happens if sqrt of scalar


        //In case of loops: if ssa comes from outside loop, we can not update in place
        //if the ssa comes from inside the loop, we can update in place

        if(daphne::EwAddOp addOp = llvm::dyn_cast_or_null<daphne::EwAddOp>(op)) {
            llvm::outs() << "EwAddOp\n";
            for( auto r :addOp.GetInPlaceOperands()) {
                llvm::outs() << r << "\n";
            }
        }


        //get 
        llvm::outs() << "Operation: " << op->getName() << "\n";

        if (daphne::InPlaceable opWithInterface = llvm::dyn_cast_or_null<daphne::InPlaceable>(op)) {
           llvm::outs() << "InPlaceOpInterface\n";
           //std::vector<int> inPlaceOperands = opWithInterface.GetInPlaceOperands();
        }

        /*
        if (op->hasTrait<mlir::OpTrait::InPlaceOperands<[0]>::Impl>()) {
            llvm::outs() << "InPlaceOperand\n";
        }
        */

        if (op->hasTrait<mlir::OpTrait::InPlaceUnaryOp>()) {

            llvm::outs() << "InPlaceUnaryOp\n";
        
            //Unary operations have only one operand and starts with 0
            auto value = op->getOperand(0);

            //Simple case
            if (value.hasOneUse() || !hasAnyUseAfterCurrentOp(op)) {
                //theoretcially the Result and Operand can be Frame to Matrix, or Matrix to Frame, Frame to Frame, Matrix to Matrix
                if (isValidType<mlir::Value>(value) && isValidType<mlir::OpResult>(op->getResult(0))) {
                        inPlaceAttr = mlir::daphne::InPlaceEnum::LHS;
                }
            }
        }
        else if (op->hasTrait<mlir::OpTrait::InPlaceBinaryOp>()) {

            bool lhsQualifies = isValidType<mlir::Value>(op->getOperand(0)) && !hasAnyUseAfterCurrentOp(op, true);
            bool rhsQualifies = isValidType<mlir::Value>(op->getOperand(1)) && !hasAnyUseAfterCurrentOp(op, false);

            if (lhsQualifies && rhsQualifies) {
                inPlaceAttr = mlir::daphne::InPlaceEnum::BOTH;
            }
            else if (lhsQualifies) {
                inPlaceAttr = mlir::daphne::InPlaceEnum::LHS;
            }
            else if (rhsQualifies) {
                inPlaceAttr = mlir::daphne::InPlaceEnum::RHS;
            }
        }

        if(inPlaceAttr != mlir::daphne::InPlaceEnum::NONE) {
            OpBuilder builder(op);
            op->setAttr("updateInPlace", builder.getI64IntegerAttr(static_cast<int64_t>(inPlaceAttr)));
            //op->setAttr("updateInPlace", mlir::daphne::UpdateInPlaceAttr::get(builder.getContext(), operandUpdateInPlaceAttr)
        }

    });
    llvm::outs() << "\033[0m";
}

std::unique_ptr<Pass> daphne::createFlagUpdateInPlacePass() {
    return std::make_unique<FlagUpdateInPlacePass>();
}