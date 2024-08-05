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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VT, class DTShape>
struct FillShape {
    static void apply(DTRes *& res, VT arg, const DTShape * shape, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VT, class DTShape>
void fillShape(DTRes *& res, VT arg, const DTShape * shape, DCTX(ctx)) {
    FillShape<DTRes, VT, DTShape>::apply(res, arg, shape, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct FillShape<DenseMatrix<VT>, VT, DenseMatrix<int64_t>> {
    static void apply(DenseMatrix<VT> *& res, VT arg, const DenseMatrix<int64_t> * shape, DCTX(ctx)) {

        const size_t numRows = shape->getNumRows();
        const size_t numCols = shape->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, arg == 0);

        if(arg != 0) {
            VT *valuesRes = res->getValues();
            for(auto i = 0ul; i < res->getNumItems(); ++i)
                valuesRes[i] = arg;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct FillShape<Matrix<VT>, VT, Matrix<int64_t >> {
    static void apply(DenseMatrix<VT> *& res, VT arg, const Matrix<int64_t> * shape, DCTX(ctx)) {

        const size_t numRows = shape->getNumRows();
        const size_t numCols = shape->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, arg == 0);

        if (arg != 0) {
            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r)
                for (size_t c = 0; c < numCols; ++c)
                    res->append(r, c, arg);
            res->finishAppend();
        }
    }
};