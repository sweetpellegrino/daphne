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

#ifndef SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLETRAITS_TD
#define SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLETRAITS_TD

include "mlir/IR/OpBase.td"

//Element-wise: 1 input element is mapped to 1 output element
//Output of cell (i,j) is only computationally dependent on the corresponding input (i,j)
def VectorElementWise : NativeOpTrait<"VectorElementWise">;

//Transpose: swapping rows and cols of a matrix
//The resulting matrix needs to be combined with swapped index??
def VectorTranspose : NativeOpTrait<"VectorTranspose">;

//Complex computation of 1 output element from multiple input elements of different matrices
//Corresponding rule-book could be used:
//Inputs: left-hand side (lhs) and right-hand side (rhs)
//Output: Single matrix
//...
def VectorMatMul : NativeOpTrait<"VectorMatMul">;

//Reduction: reduction of matrix to 1 output element or to vector
//The result is reduced after an additional aggregation of intermediate results
def VectorReduction : NativeOpTrait<"VectorReduction">;


#endif //SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLETRAITS_TD