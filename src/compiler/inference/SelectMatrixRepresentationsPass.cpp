/*
 * Copyright 2021 The DAPHNE Consortium
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

#include "api/cli/DaphneUserConfig.h"
#include "llvm/Support/raw_ostream.h"
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <stdexcept>
#include <memory>
#include <string>
#include <vector>
#include <utility>

using namespace mlir;

class SelectMatrixRepresentationsPass : public PassWrapper<SelectMatrixRepresentationsPass, OperationPass<func::FuncOp>> {

public:
    void runOnOperation() override {
        func::FuncOp f = getOperation();

        const DaphneUserConfig& l_cfg = this->cfg;

        std::function<void(mlir::Operation*)> walkOp;
        walkOp = [&](mlir::Operation *op) {
            if(returnsKnownProperties(op)) {
                const bool isScfOp = op->getDialect() == op->getContext()->getOrLoadDialect<scf::SCFDialect>();

                // ----------------------------------------------------------------
                // Handle all non-SCF operations
                // ----------------------------------------------------------------
                if(!isScfOp) {
                    // Set the matrix representation for all result types
                    for(auto res : op->getResults()) {
                        if(auto matTy = res.getType().dyn_cast<daphne::MatrixType>()) {
                            const double sparsity = matTy.getSparsity();
                            // TODO: set threshold by user

                            auto fileLoc = op->getLoc().dyn_cast<mlir::FileLineColLoc>();
                            int isOdd = fileLoc.getLine() % 2;

                            daphne::MatrixRepresentation matrixRep;
                            switch (l_cfg.force_sparse) {
                                case SPARSE_COMB::CSR_CSR:
                                    matrixRep = daphne::MatrixRepresentation::SparseCSR;
                                    break;
                                case SPARSE_COMB::CSC_CSC:
                                    matrixRep = daphne::MatrixRepresentation::SparseCSC;
                                    break;
                                case SPARSE_COMB::CSR_CSC:
                                    if (!isOdd) {
                                        matrixRep = daphne::MatrixRepresentation::SparseCSR;
                                        }
                                    else {
                                        matrixRep = daphne::MatrixRepresentation::SparseCSC;
                                        }
                                    break;
                                case SPARSE_COMB::CSC_CSR:
                                    if (!isOdd) {
                                        matrixRep = daphne::MatrixRepresentation::SparseCSC;
                                        }
                                    else {
                                        matrixRep = daphne::MatrixRepresentation::SparseCSR;
                                        }
                                    break;
                                default:
                                    matrixRep = daphne::MatrixRepresentation::Dense;
                                    break;
                            }
                            res.setType(matTy.withRepresentation(matrixRep));
                            std::string outType;
                            if (matrixRep == daphne::MatrixRepresentation::SparseCSR) {
                                outType = "SparseCSR";
                            } else if (matrixRep == daphne::MatrixRepresentation::SparseCSC) {
                                outType = "SparseCSC";
                            } else {
                                outType = "Dense";
                            }
                            llvm::errs() << op->getName() << " at line: " << fileLoc.getLine() << " and chosen matrix rep " << outType << "\n";
    //                        if(sparsity < 0.1) {
    //                            res.setType(matTy.withRepresentation(daphne::MatrixRepresentation::SparseCSR));
    //                        }
                        }
                    }
                }
                // TODO For later: Don't duplicate the special treatment of SCF
                // ops from InferencePass.
                // ----------------------------------------------------------------
                // Special treatment for some SCF operations
                // ----------------------------------------------------------------
                else if(auto whileOp = llvm::dyn_cast<scf::WhileOp>(op)) {
                    Block &beforeBlock = whileOp.getBefore().front();
                    Block &afterBlock = whileOp.getAfter().front();
                    // Transfer the WhileOp's operand types to the block arguments
                    // and results to fulfill constraints on the WhileOp.
                    for(size_t i = 0 ; i < whileOp.getNumOperands() ; i++) {
                        Type t = whileOp->getOperand(i).getType();
                        beforeBlock.getArgument(i).setType(t);
                        afterBlock.getArgument(i).setType(t);
                        whileOp.getResult(i).setType(t);
                    }
                    // Continue the walk on both blocks of the WhileOp. We trigger
                    // this explicitly, since we need to do something afterwards.
                    beforeBlock.walk<WalkOrder::PreOrder>(walkOp);
                    afterBlock.walk<WalkOrder::PreOrder>(walkOp);

                    // Check if the infered matrix representations match the required result representations.
                    // This is not the case if, for instance, the representation of some
                    // variable written in the loop changes. The WhileOp would also
                    // check this later during verification, but here, we want to
                    // throw a readable error message.
                    Operation *yieldOp = afterBlock.getTerminator();
                    for(size_t i = 0 ; i < whileOp.getNumOperands() ; i++) {
                        Type yieldedTy = yieldOp->getOperand(i).getType();
                        Type resultTy = op->getResult(i).getType();
                        if(yieldedTy != resultTy)
                            throw std::runtime_error(
                                "the representation of a matrix must not be "
                                "changed within the body of a while-loop"
                            );
                    }
                    // Tell the walker to skip the descendants of the WhileOp, we
                    // have already triggered a walk on them explicitly.
                    return WalkResult::skip();
                }
                else if(auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
                    Block &block = forOp.getRegion().front();
                    const size_t numIndVars = forOp.getNumInductionVars();
                    // Transfer the ForOp's operand types to the block arguments
                    // and results to fulfill constraints on the ForOp.
                    for(size_t i = 0 ; i < forOp.getNumIterOperands() ; i++) {
                        Type t = forOp.getIterOperands()[i].getType();
                        block.getArgument(i + numIndVars).setType(t);
                        forOp.getResult(i).setType(t);
                    }
                    // Continue the walk on the body block of the ForOp. We trigger
                    // this explicitly, since we need to do something afterwards.
                    block.walk<WalkOrder::PreOrder>(walkOp);
                    // Check if the infered matrix representations match the required result representations.
                    // This is not the case if, for instance, the representation of some
                    // variable written in the loop changes. The ForOp would also
                    // check this later during verification, but here, we want to
                    // throw a readable error message.
                    Operation *yieldOp = block.getTerminator();
                    for(size_t i = 0 ; i < forOp.getNumIterOperands() ; i++) {
                        Type yieldedTy = yieldOp->getOperand(i).getType();
                        Type resultTy = op->getResult(i).getType();
                        if(yieldedTy != resultTy)
                            throw std::runtime_error(
                                "the representation of a matrix must not be "
                                "changed within the body of a for-loop"
                            );
                    }
                    // Tell the walker to skip the descendants of the ForOp, we
                    // have already triggered a walk on them explicitly.
                    return WalkResult::skip();
                }
                else if(auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
                    // Walk the then/else blocks first. We need the inference on
                    // them before we can do anything about the IfOp itself.
                    ifOp.thenBlock()->walk<WalkOrder::PreOrder>(walkOp);
                    ifOp.elseBlock()->walk<WalkOrder::PreOrder>(walkOp);
                    // Check if the yielded matrix representations are the same in both
                    // branches. The IfOp would also check this later during
                    // verification, but here, we want to throw a readable error
                    // message.
                    // Additionally, we set the result types of the IfOp here.
                    scf::YieldOp thenYield = ifOp.thenYield();
                    scf::YieldOp elseYield = ifOp.elseYield();
                    for(size_t i = 0 ; i < ifOp.getNumResults() ; i++) {
                        Type thenTy = thenYield->getOperand(i).getType();
                        Type elseTy = elseYield->getOperand(i).getType();
                        if(thenTy != elseTy)
                            throw std::runtime_error(
                                "a matrix must not be assigned two values of "
                                "different representations in then/else branches"
                            );
                        ifOp.getResult(i).setType(thenTy);
                    }
                    // Tell the walker to skip the descendants of the IfOp, we
                    // have already triggered a walk on them explicitly.
                    return WalkResult::skip();
                }
            }
            // Continue the walk normally.
            return WalkResult::advance();
        };
        f.walk<WalkOrder::PreOrder>(walkOp);
        // infer function return types
        // TODO: cast for UDFs
        f.setType(FunctionType::get(&getContext(),
            f.getFunctionType().getInputs(),
            f.getBody().back().getTerminator()->getOperandTypes()));
    }

    explicit SelectMatrixRepresentationsPass(const DaphneUserConfig& cfg) : cfg(cfg) { }
    const DaphneUserConfig& cfg;
 

    StringRef getArgument() const final { return "select-matrix-representations"; }
    StringRef getDescription() const final { return "TODO"; }

    static bool returnsKnownProperties(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type rt) {
            if(auto mt = rt.dyn_cast<daphne::MatrixType>())
                return mt.getSparsity() != -1.0;
            return false;
        });
    }
};

std::unique_ptr<Pass> daphne::createSelectMatrixRepresentationsPass(const DaphneUserConfig& cfg){
    return std::make_unique<SelectMatrixRepresentationsPass>(cfg);
}
