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

template<class DTRes, typename DTArg>
void fusedVectorizedPipeline(DTRes res, DTArg row) {
    return ewUnaryMat<DTRes, DTArg>(UnaryOpCode::SQRT, res, row, nullptr);
}

// ****************************************************************************
// ewBinaryMat
// ****************************************************************************

template<class DT,typename VT>
void generateBinaryMatrices(DT *& m1, DT *& m2, size_t numRows, size_t numCols, VT val1, VT val2) {
    fillMatrix<DT, VT>(m1, numRows, numCols, val1);
    fillMatrix<DT, VT>(m2, numRows, numCols, val2);
}

TEMPLATE_PRODUCT_TEST_CASE("VectorizedPipeline - Fused Ops", TAG_SPARSEMATRIX_BENCH, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    using VT = typename DT::VT;

    BENCHMARK_ADVANCED("MAIN") (Catch::Benchmark::Chronometer meter) {
        DT* m1 = nullptr;
        DT* m2 = nullptr;
        generateBinaryMatrices<DT, VT>(m1, m2, 5000, 5000, VT(1), VT(2));
        DT* res = nullptr;

        meter.measure([&m1, &m2, &res]() {
            return ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, res, m1, m2, nullptr);
        });

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(m2);
        DataObjectFactory::destroy(res);
    };
}

TEMPLATE_PRODUCT_TEST_CASE("VectorizedPipeline - Sequential Pipes", TAG_SPARSEMATRIX_BENCH, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    using VT = typename DT::VT;

    BENCHMARK_ADVANCED("MAIN") (Catch::Benchmark::Chronometer meter) {
        DT* m1 = nullptr;
        DT* m2 = nullptr;
        generateBinaryMatrices<DT, VT>(m1, m2, 5000, 5000, VT(1), VT(2));
        DT* res = nullptr;

        meter.measure([&m1, &m2, &res]() {
            return ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, res, m1, m2, nullptr);
        });

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(m2);
        DataObjectFactory::destroy(res);
    };
}