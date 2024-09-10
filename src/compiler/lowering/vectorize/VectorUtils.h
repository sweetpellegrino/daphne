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

#pragma once
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include <cstddef>
#include <iostream>
#include <fstream>
#include <llvm/ADT/SmallVector.h>
#include <stack>
#include <vector>
#include <algorithm>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

using VectorIndex = std::size_t;

struct VectorUtils {

    static bool matchingVectorSplitCombine(const mlir::daphne::VectorSplit split, const mlir::daphne::VectorCombine combine) {
        mlir::daphne::VectorCombine _operandCombine;
        switch (split) {
            case mlir::daphne::VectorSplit::ROWS:
                _operandCombine = mlir::daphne::VectorCombine::ROWS;
                break;
            case mlir::daphne::VectorSplit::COLS:
                _operandCombine = mlir::daphne::VectorCombine::COLS;
                break;
            default:
                // No matching split/combine; basically resulting in separate pipelines
                return false;
        }
        if (combine == _operandCombine)
            return true;
        return false;
    }

    //-----------------------------------------------------------------
    // Printing graph/pipelines
    //-----------------------------------------------------------------

    static void printGraph(std::vector<mlir::Operation*> leafOps, std::string filename) {
        std::stack<mlir::Operation*> stack;
        std::ofstream dot(filename);
        if (!dot.is_open()) {
            throw std::runtime_error("test");
        }

        dot << "digraph G {\n";
        for (auto s : leafOps) {
            stack.push(s);
            stack.push(s);
        }

        std::vector<mlir::Operation*> visited;

        while (!stack.empty()) {
            auto op = stack.top(); stack.pop();
            if(std::find(visited.begin(), visited.end(), op) != visited.end()) {
                continue;
            }
            visited.push_back(op);

            auto v = llvm::dyn_cast<mlir::daphne::Vectorizable>(op);
            if (!v) 
                continue;

            for (unsigned i = 0; i < v->getNumOperands(); ++i) {
                mlir::Value e = v->getOperand(i);
                auto defOp = e.getDefiningOp();
                if (llvm::isa<mlir::daphne::MatrixType>(e.getType()) && llvm::isa<mlir::daphne::Vectorizable>(defOp)) {
                    dot << "\"" << defOp->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(defOp) << "\"";
                    dot << " -> ";
                    dot << "\"" << op->getName().getStringRef().str() << "+" << std::hex << reinterpret_cast<uintptr_t>(op) << "\" [label=\"" << i << "\"];\n";
                    stack.push(defOp);
                }
            }
        }
        dot << "}";
        dot.close();
    }

    static std::string getColor(size_t pipelineId) {
        std::vector<std::string> colors = {"tomato", "lightgreen", "lightblue", "plum1", "navajowhite1", "seashell", "hotpink",
                                        "lemonchiffon", "firebrick1", "ivory2", "khaki1", "lightcyan", "olive", "yellow",
                                        "maroon", "violet", "mistyrose2"};
        return colors[pipelineId % colors.size()];
    }

    static void printPipelines(std::vector<mlir::Operation*> &vectOps, std::map<mlir::Operation*, size_t> &operationToPipelineIx, std::vector<VectorIndex> &dIxs, std::string filename) {
        std::ofstream outfile(filename);

        outfile << "digraph G {" << std::endl;

        std::map<mlir::Operation*, std::string> opToNodeName;

        for (size_t i = 0; i < vectOps.size(); ++i) {
            std::string nodeName = "node" + std::to_string(i);
            opToNodeName[vectOps[i]] = nodeName;

            size_t pipelineId = operationToPipelineIx.at(vectOps[i]);
            std::string color = VectorUtils::getColor(pipelineId);

            outfile << nodeName << " [label=\"" << vectOps[i]->getName().getStringRef().str() << "\\npIx: " << pipelineId << ", dIx: " << dIxs[i] << "\", fillcolor=" << color << ", style=filled];" << std::endl;
        }

        for (size_t i = 0; i < vectOps.size(); ++i) {
            mlir::Operation* op = vectOps[i];
            auto consumerPipelineIx = operationToPipelineIx.at(op);

            for (auto operandValue : op->getOperands()) {
                mlir::Operation* operandOp =operandValue.getDefiningOp();
                auto it = operationToPipelineIx.find(operandOp);

                if (it != operationToPipelineIx.end()) {
                    auto producerPipeplineIx = it->second;
                    outfile << opToNodeName[operandOp] << " -> " << opToNodeName[op];

                    if (producerPipeplineIx != consumerPipelineIx) {
                        outfile << " [style=dotted]";
                    }
                    outfile << ";" << std::endl;
                } 
            }
        }
        outfile << "}" << std::endl;
    }
};