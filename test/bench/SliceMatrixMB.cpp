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

#include "run_tests.h"
#include "runtime/local/kernels/SliceCol.h"
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/CSCMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/EwBinaryObjSca.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/Transpose.h>
#include <runtime/local/kernels/Reverse.h>
#include <runtime/local/kernels/Fill.h>

#include <tags.h>
#include <catch.hpp>
#include <vector>
#include <runtime/local/kernels/RandMatrix.h>
#include <papi.h>


// TODO: use Fill kernel
template<class DT, typename VT>
void fillMatrix(DT *& m, size_t numRows, size_t numCols, VT val) {

    m = DataObjectFactory::create<DT>(numRows, numCols, false);

    VT * values = m->getValues();
    const size_t numCells = numRows * numCols;

    // Fill the matrix with ones of the respective value type.
    for(size_t i = 0; i < numCells; i++)
        values[i] = 1;

}

// ****************************************************************************
// ewBinaryMat
// ****************************************************************************

template<class DT,typename VT>
void generateBinaryMatrices(DT *& m1, DT *& m2, size_t numRows, size_t numCols, VT val1, VT val2) {
    fillMatrix<DT, VT>(m1, numRows, numCols, val1);
    fillMatrix<DT, VT>(m2, numRows, numCols, val2);
}

TEMPLATE_PRODUCT_TEST_CASE("", TAG_SLICEMATRIX_BENCH, (CSRMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto dctx = setupContextAndLogger();

    DT* m1 = nullptr;
    randMatrix(m1, 5, 5, 1.0, 5.0, 0.3, -1, nullptr);
    m1->print(std::cout); 

    DT* res = nullptr;
    sliceCol<DT, DT, size_t>(res, m1, 0, 3, nullptr);
    m1->print(std::cout); 
    res->print(std::cout); 
}