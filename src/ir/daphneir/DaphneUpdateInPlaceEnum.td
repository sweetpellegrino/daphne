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

#ifndef SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEENUM_TD
#define SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEENUM_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def INPLACE_NONE : I64EnumAttrCase<"NONE", 0>;
def INPLACE_LHS : I64EnumAttrCase<"LHS", 1>;
def INPLACE_RHS : I64EnumAttrCase<"RHS", 2>;
def INPLACE_BOTH : I64EnumAttrCase<"BOTH", 3>;

def InPlaceAttr : I64EnumAttr<"InPlace", "", [INPLACE_NONE, INPLACE_LHS, INPLACE_RHS, INPLACE_BOTH]> {
    let cppNamespace = "::mlir::daphne";
}

#endif // SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEENUM_TD
