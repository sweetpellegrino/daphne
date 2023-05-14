//===- TestPrintDefUse.cpp - Passes to illustrate the IR def-use chains ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <string>
#include <iostream>


using namespace mlir;

/// This pass illustrates the IR def-use chains through printing.
struct TestPrintDefUsePass
    : public PassWrapper<TestPrintDefUsePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintDefUsePass)

  StringRef getArgument() const final { return "test-print-defuse"; }
  StringRef getDescription() const final { return "Test various printing."; }
  void runOnOperation() override {
    // Recursively traverse the IR nested under the current operation and print
    // every single operation and their operands and users.
    getOperation()->walk([](Operation *op) {
      op->dump();
      llvm::errs() << "//" << "Visiting op '" << op->getName() << "' with "
                   << op->getNumOperands() << " operands:\n";

      // Print information about the producer of each of the operands.
      for (Value operand : op->getOperands()) {
        if (Operation *producer = operand.getDefiningOp()) {
          llvm::errs() << "//" << "  - Operand produced by operation '"
                       << producer->getName() << "'\n";
        } else {
          // If there is no defining op, the Value is necessarily a Block
          // argument.
          auto blockArg = cast<BlockArgument>(operand);
          llvm::errs() << "//" << "  - Operand produced by Block argument, number "
                       << blockArg.getArgNumber() << "\n";
        }
      }

      // Print information about the user of each of the result.
      llvm::errs() << "//" << "Has " << op->getNumResults() << " results:\n";
      for (const auto &indexedResult : llvm::enumerate(op->getResults())) {
        Value result = indexedResult.value();
        llvm::errs() << "//" << "  - Result " << indexedResult.index();
        if (result.use_empty()) {
          llvm::errs() << " has no uses\n";
          continue;
        }
        if (result.hasOneUse()) {
          llvm::errs() << " has a single use: \n";
        } else {
          llvm::errs() << " has "
                       << std::distance(result.getUses().begin(),
                                        result.getUses().end())
                       << " uses:\n";
        }
        for (Operation *userOp : result.getUsers()) {
          llvm::errs() << "//" << "    - " << userOp->getName() << "\n";
        }
      }
    });
  }
};

std::unique_ptr<Pass> daphne::createTestPrintDefUsePass() {
    return std::make_unique<TestPrintDefUsePass>();
}