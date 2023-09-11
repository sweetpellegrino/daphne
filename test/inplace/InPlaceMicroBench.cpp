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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>

#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/EwBinaryObjSca.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/Transpose.h>
#include <runtime/local/kernels/Fill.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>
#include <cstdint>


// ****************************************************************************
// ewBinaryMat
// ****************************************************************************

// TODO: use  
template<class DT,typename VT>
void fillMatrix(DT *& m, size_t numRows, size_t numCols, VT val) {
    
    DT * m = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
    
    VT * values = m->getValues();
    const size_t numCells = numRows * numCols;
    
    // Fill the matrix with ones of the respective value type.
    for(size_t i = 0; i < numCells; i++)
        values[i] = VT(1);
}

TEMPLATE_PRODUCT_TEST_CASE("ewBinaryMat - In Place - Bench", TAG_INPLACE_BENCH, (DenseMatrix), (uint32_t)) {
    using DT = TestType;
    using VT = DT::ValueType;
    // Measure the execution time of myFunctionToBenchmark using Catch's benchmarking macros
    DT* m1 = nullptr;
    DT* m2 = nullptr;

    DT* res = nullptr;

    BENCHMARK("My Function Benchmark") {
        ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, res, m1, m2, false, false, nullptr);
    };
}