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

#include "runtime/local/datastructures/DenseMatrix.h"
#include <ir/daphneir/Daphne.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>

#include <stdexcept>
#include <string>
#include <iostream>
#include <type_traits>

class InPlaceUtils {
public:

    //correct type?
    template<typename VTLhs, typename VTRhs>
    static bool isValidType(const DenseMatrix<VTLhs> *res, const DenseMatrix<VTRhs> *arg) {
        if(res->getNumCols() == arg->getNumCols() &&
           res->getNumRows() == arg->getNumRows() &&
           std::is_same_v<VTLhs, VTRhs>
        ) {
            return true;
        }
        return false;
    }

    template<typename VTArg, typename... Args>
    static DenseMatrix<VTArg>* getResultsPointer(DenseMatrix<VTArg> *arg, bool hasFutureUseArg, Args... args) {
        if (!hasFutureUseArg) {
            if (arg->getRefCounter() == 1 && arg->getValuesUseCount() == 1) {
                arg->increaseRefCounter();
                return arg;
            }
        }

        if constexpr (sizeof...(Args) == 0) {
            return nullptr;
        } else {
            return getResultsPointer(args...);
        }
    }

};

//was muss berücksichtigt werden?
//1. Operanden und Ergebnis müssen den gleichen Typ haben
//1.1 darf nach gewissen Regeln trotzdem erlaubt sein, z.b. transpose
//2. Operanden dürfen nicht nach dem aktuellen Op verwendet werden



#endif //SRC_RUNTIME_LOCAL_UTILS_RUNTIMEUTILS_H