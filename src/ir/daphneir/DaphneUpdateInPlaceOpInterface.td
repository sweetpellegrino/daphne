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

/*def INPLACE_ONE : I64EnumAttrCase<"ONE", 0>;
def INPLACE_ARG : I64EnumAttrCase<"ARG", 0>;
def INPLACE_LHS : I64EnumAttrCase<"LHS", 0>;
def INPLACE_TWO : I64EnumAttrCase<"TWO", 1>;
def INPLACE_RHS : I64EnumAttrCase<"RHS", 1>;
def INPLACE_THREE : I64EnumAttrCase<"THREE", 2>;
def INPLACE_FOUR : I64EnumAttrCase<"FOUR", 3>;
def INPLACE_FIVE : I64EnumAttrCase<"FIVE", 4>;
def INPLACE_SIX : I64EnumAttrCase<"SIX", 5>;
def INPLACE_SEVEN : I64EnumAttrCase<"SEVEN", 6>;
def INPLACE_EIGHT : I64EnumAttrCase<"EIGHT", 7>;
def INPLACE_NINE : I64EnumAttrCase<"NINE", 8>;

def InPlaceOperand : I64EnumAttr<"InPlace", "", [INPLACE_ONE, INPLACE_TWO, INPLACE_THREE, INPLACE_FOUR, INPLACE_FIVE, INPLACE_SIX, INPLACE_SEVEN, INPLACE_EIGHT, INPLACE_NINE]> {
    let cppNamespace = "::mlir::daphne";
}*/

def INPLACE_NONE : I64EnumAttrCase<"NONE", -1>;
def INPLACE_ARG : I64EnumAttrCase<"ARG", 1>;
def INPLACE_LHS : I64EnumAttrCase<"LHS", 2>;
def INPLACE_RHS : I64EnumAttrCase<"RHS", 3>;
def INPLACE_BOTH : I64EnumAttrCase<"BOTH", 4>;

def InPlaceEnum : I64EnumAttr<"InPlaceEnum", "", [INPLACE_NONE, INPLACE_ARG, INPLACE_LHS, INPLACE_RHS, INPLACE_BOTH]> {
    let cppNamespace = "::mlir::daphne";
}

def InPlaceUnaryOp : NativeOpTrait<"InPlaceUnaryOp">;
def InPlaceBinaryOp : NativeOpTrait<"InPlaceBinaryOp">;

class InPlaceOperands<list<int> positions> : ParamNativeOpTrait<"InPlaceOperands", !cast<string>(!head(positions))>;

//https://github.com/tensorflow/tensorflow/blob/fd8d0c80373f6a011cfab909ee23ce4b9c42e0c2/tensorflow/compiler/mlir/lite/ir/tfl_op_interfaces.td#L59
def InPlaceOpInterface : OpInterface<"InPlaceable"> {
    let description = [{
        Interface for ops that are stateful and need to identify stateful operands.

        Stateful operands correspond to TF's variables semantics. An op that has 1
        or more stateful operands is a stateful op.
    }];
  
    let methods = [
        InterfaceMethod<
        [{Returns the indices of potential candidates of operands that should allow for update in place.}],
        "std::vector<int>", "GetInPlaceOperands", (ins)
        >,
    ];
}

def TFL_StatefulOp : OpInterface<"StatefulOpInterface"> {
  let description = [{
    Interface for ops that are stateful and need to identify stateful operands.

    Stateful operands correspond to TF's variables semantics. An op that has 1
    or more stateful operands is a stateful op.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the indices of stateful operands.}],
      "std::vector<int>", "GetStatefulOperands", (ins)
    >,
  ];
}


//class MyParametricTrait<int... prop> : NativeOpTrait<"MyParametricTrait", !cast<string>(!head(parameters))>;

//def MyParametricTrait2 : NativeOpTrait<"MyParametricTrait2">;

#endif // SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEENUM_TD
