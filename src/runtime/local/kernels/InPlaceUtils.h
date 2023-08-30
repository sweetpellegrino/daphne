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


#ifndef SRC_RUNTIME_LOCAL_UTILS_RUNTIMEUTILS_H
#define SRC_RUNTIME_LOCAL_UTILS_RUNTIMEUTILS_H

#include <ir/daphneir/Daphne.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>

#include <stdexcept>
#include <string>

class InPlaceUtils {
public:
    //check type of operand is equal to type of result
    static bool isValidType(mlir::Value arg, mlir::Value res) {
        if (arg.getType() != res.getType()) {
            return false;
        }
        return true;
    }

};

//was muss berücksichtigt werden?
//1. Operanden und Ergebnis müssen den gleichen Typ haben
//1.1 darf nach gewissen Regeln trotzdem erlaubt sein, z.b. transpose
//2. Operanden dürfen nicht nach dem aktuellen Op verwendet werden



#endif //SRC_RUNTIME_LOCAL_UTILS_RUNTIMEUTILS_H