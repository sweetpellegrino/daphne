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
#include "runtime/local/kernels/SliceRow.h"
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

// ****************************************************************************
// ewBinaryMat
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE("", TAG_SLICEMATRIX_BENCH, (CSRMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    //auto dctx = setupContextAndLogger();

    int numRows = 1000;
    int numCols = 1000;

    BENCHMARK_ADVANCED("sliceCol (sym)") (Catch::Benchmark::Chronometer meter) {
        DT* m1 = nullptr;
        randMatrix(m1, numRows, numCols, 1.0, 2.0, 0.3, -1, nullptr);
        DT* res = nullptr;

        meter.measure([&m1, &res]() {
            return sliceCol<DT, DT, size_t>(res, m1, 101, 599, nullptr); 
        });

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(res);
    };

    BENCHMARK_ADVANCED("sliceRow (sym)") (Catch::Benchmark::Chronometer meter) {
        DT* m1 = nullptr;
        randMatrix(m1,numRows, numCols, 1.0, 2.0, 0.3, -1, nullptr);
        DT* res = nullptr;

        meter.measure([&m1, &res]() {
            return sliceRow<DT, DT, size_t>(res, m1, 101, 599, nullptr); 
        });

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(res);
    };

    BENCHMARK_ADVANCED("sliceCol (sym) tiny slice") (Catch::Benchmark::Chronometer meter) {
        DT* m1 = nullptr;
        randMatrix(m1, numRows, numCols, 1.0, 2.0, 0.3, -1, nullptr);
        DT* res = nullptr;

        meter.measure([&m1, &res]() {
            return sliceCol<DT, DT, size_t>(res, m1, 1, 5, nullptr); 
        });

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(res);
    };

    BENCHMARK_ADVANCED("sliceCol (sym) small slice") (Catch::Benchmark::Chronometer meter) {
        DT* m1 = nullptr;
        randMatrix(m1,numRows, numCols, 1.0, 2.0, 0.3, -1, nullptr);
        DT* res = nullptr;

        meter.measure([&m1, &res]() {
            return sliceCol<DT, DT, size_t>(res, m1, 10, 50, nullptr); 
        });

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(res);
    };

    BENCHMARK_ADVANCED("sliceCol (sym) big slice") (Catch::Benchmark::Chronometer meter) {
        DT* m1 = nullptr;
        randMatrix(m1,numRows, numCols, 1.0, 2.0, 0.3, -1, nullptr);
        DT* res = nullptr;

        meter.measure([&m1, &res]() {
            return sliceCol<DT, DT, size_t>(res, m1, 0, 1000, nullptr); 
        });

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(res);
    };

}