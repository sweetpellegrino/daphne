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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_GENERATOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_GENERATOR_H

#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/GenMetaData.h"
#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Structure.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes>
struct Generator {
    static void apply(DTRes *& res, size_t numRows, size_t numCols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void generator(DTRes *& res, size_t numRows, size_t numCols, DCTX(ctx)) {
    Generator<DTRes>::apply(res, numRows, numCols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct Generator<DenseMatrix<int64_t>> {
    static void apply(DenseMatrix<int64_t> *& res, size_t numRows, size_t numCols, DCTX(ctx)) {

        std::shared_ptr<int64_t[]> ptr(nullptr);
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<int64_t>>(numRows, numCols, ptr);

    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_GENERATOR_H