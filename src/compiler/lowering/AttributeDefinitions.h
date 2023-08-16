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

#ifndef SRC_COMPILER_LOWERING_ATTRIBUTEDEFINITIONS_H
#define SRC_COMPILER_LOWERING_ATTRIBUTEDEFINITIONS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include <string>

//using namespace mlir::daphne;

const std::string ATTR_HASVARIADICRESULTS = "hasVariadicResults";
const std::string ATTR_UPDATEINPLACE_KEY = "updateInPlace";

enum class ATTR_UPDATEINPLACE_TYPE {
    NONE,
    LHS,
    RHS,
    BOTH,
 };

//TODO: Define a custom trait?


#endif //SRC_COMPILER_LOWERING_ATTRIBUTEDEFINITIONS_H