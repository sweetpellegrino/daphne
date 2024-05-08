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
#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/Structure.h"
#include <cstddef>
#include <iostream>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/datagen/GenGivenVals.h>

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/CSCMatrix.h>

#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/Transpose.h>

#include <runtime/local/kernels/Fill.h>
#include <runtime/local/kernels/RandMatrix.h>
#include <runtime/local/kernels/VectorizedPipeline.h>

#include <tags.h>
#include <catch.hpp>
#include <vector>
#include <papi.h>

/*
  llvm.func @_vect1(%arg0: !llvm.ptr<ptr<ptr<i1>>>, %arg1: !llvm.ptr<ptr<i1>>, %arg2: !llvm.ptr<i1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.load %arg1 : !llvm.ptr<ptr<i1>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.getelementptr %arg1[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %4 = llvm.load %3 : !llvm.ptr<ptr<i1>>
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %7 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %7, %6 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%6, %1, %4, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %8 = llvm.load %6 : !llvm.ptr<ptr<i1>>
    %9 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%4, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.alloca %10 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %12 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %12, %11 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%11, %1, %8, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %13 = llvm.load %11 : !llvm.ptr<ptr<i1>>
    %14 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%1, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.load %arg0 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %8, %16 : !llvm.ptr<ptr<i1>>
    %17 = llvm.mlir.constant(1 : i64) : i64
    %18 = llvm.getelementptr %arg0[1] : (!llvm.ptr<ptr<ptr<i1>>>) -> !llvm.ptr<ptr<ptr<i1>>>
    %19 = llvm.load %18 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %13, %19 : !llvm.ptr<ptr<i1>>
    llvm.return
  }
*/
static void fusedVectorizedPipeline(DenseMatrix<double> ***res, Structure **rows, DaphneContext *ctx) {
    DenseMatrix<double> *& _res = *res[0];
    DenseMatrix<double> *_rows = reinterpret_cast<DenseMatrix<double>*>(rows[0]);

    DenseMatrix<double> * _localRes = nullptr;

    transpose<DenseMatrix<double>, DenseMatrix<double>>(_localRes, _rows, ctx);
    ewUnaryMat<DenseMatrix<double>, DenseMatrix<double>>(UnaryOpCode::SQRT, _res, _localRes, ctx);
}

static void transposeVectorizedPipeline(DenseMatrix<double> ***res, Structure **rows, DaphneContext *ctx) {
    DenseMatrix<double> *& _res = *res[0];
    DenseMatrix<double> *_rows = reinterpret_cast<DenseMatrix<double>*>(rows[0]);
    transpose<DenseMatrix<double>, DenseMatrix<double>>(_res, _rows, ctx);
}

static void sqrtVectorizedPipeline(DenseMatrix<double> ***res, Structure **rows, DaphneContext *ctx) {
    DenseMatrix<double> *& _res = *res[0];
    DenseMatrix<double> *_rows = reinterpret_cast<DenseMatrix<double>*>(rows[0]);
    ewUnaryMat<DenseMatrix<double>, DenseMatrix<double>>(UnaryOpCode::SQRT, _res, _rows, ctx);
}

int numRows = 2050;
int numCols = 1950;

TEMPLATE_PRODUCT_TEST_CASE("VectorizedPipeline - Sequential Vec Ops", TAG_VECTORIZED_BENCH, (DenseMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto dctx = setupContextAndLogger();
    DT* m1 = nullptr;
    randMatrix(m1, numRows, numCols, 1.0, 5.0, 1.0, -1, nullptr);

    BENCHMARK_ADVANCED("MAIN") (Catch::Benchmark::Chronometer meter) {

        // Set up Data
        //int numRows = 10;
        //int numCols = 5;

        // Set up VectorizedPipeline (transpose)
        size_t t_numOutputs = 1;
        size_t t_numInputs = 1;

        DT** t_outputs = new DT*[t_numOutputs]; 
        t_outputs[0] = nullptr;; 
        
        bool* t_isScalar = new bool[t_numInputs];
        t_isScalar[0] = false;

        Structure** t_inputs = new Structure*[t_numInputs];
        t_inputs[0] = m1; 

        int64_t t_outRows[1] = {numCols}; // <- flipped 
        int64_t t_outCols[1] = {numRows}; // <- flipped 
        int64_t t_splits[1] = {1};
        int64_t t_combines[1] = {2};

        size_t t_numFuncs = 1;
        void (*t_funPtrs[1])(DT ***, Structure **, DaphneContext*) = {
            transposeVectorizedPipeline
        };
        void **t_fun = reinterpret_cast<void**>(t_funPtrs);

        // Set up VectorizedPipeline (sqrt)
        size_t sqrt_numOutputs = 1;
        size_t sqrt_numInputs = 1;

        DT** sqrt_outputs = new DT*[sqrt_numOutputs];
        sqrt_outputs[0] = nullptr;;
        
        bool* sqrt_isScalar = new bool[sqrt_numInputs];
        sqrt_isScalar[0] = false;

        Structure** sqrt_inputs = new Structure*[sqrt_numInputs];
        sqrt_inputs[0] = nullptr;

        int64_t sqrt_outRows[1] = {numCols}; // <- flipped 
        int64_t sqrt_outCols[1] = {numRows}; // <- flipped 
        int64_t sqrt_splits[1] = {1};
        int64_t sqrt_combines[1] = {1};

        size_t sqrt_numFuncs = 1;
        void (*sqrt_funPtrs[1])(DT ***, Structure **, DaphneContext*) = {
            sqrtVectorizedPipeline
        };
        void **sqrt_fun = reinterpret_cast<void**>(sqrt_funPtrs);

        meter.measure([&t_outputs, &t_numOutputs, &t_isScalar, &t_inputs, &t_numInputs, &t_outRows, &t_outCols, &t_splits, &t_combines, &t_numFuncs, &t_fun, &sqrt_outputs, &sqrt_numOutputs, &sqrt_isScalar, &sqrt_inputs, &sqrt_numInputs, &sqrt_outRows, &sqrt_outCols, &sqrt_splits, &sqrt_combines, &sqrt_numFuncs, &sqrt_fun, &dctx]() {
            
            //TRANSPOSE
            vectorizedPipeline<DT>(t_outputs, t_numOutputs, t_isScalar, t_inputs, t_numInputs, t_outRows, t_outCols, t_splits, t_combines, t_numFuncs, t_fun, dctx.get()); 
            //SQRT
            sqrt_inputs[0] = t_outputs[0];
            vectorizedPipeline<DT>(sqrt_outputs, sqrt_numOutputs, sqrt_isScalar, sqrt_inputs, sqrt_numInputs, sqrt_outRows, sqrt_outCols, sqrt_splits, sqrt_combines, sqrt_numFuncs, sqrt_fun, dctx.get()); 
            return;
            
       });

        /*std::cout << "Test" << std::endl;
        m1->print(std::cout); 
        t_outputs[0]->print(std::cout); 
        sqrt_outputs[0]->print(std::cout); 
        std::cout << "Test" << std::endl;*/
        DataObjectFactory::destroy(t_outputs[0]);
        DataObjectFactory::destroy(sqrt_outputs[0]);
    };
    DataObjectFactory::destroy(m1);
}

TEMPLATE_PRODUCT_TEST_CASE("VectorizedPipeline - Fused Ops", TAG_VECTORIZED_BENCH, (DenseMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto dctx = setupContextAndLogger();

    DT* m1 = nullptr;
    randMatrix(m1, numRows, numCols, 1.0, 5.0, 1.0, -1, nullptr);

    BENCHMARK_ADVANCED("MAIN") (Catch::Benchmark::Chronometer meter) {
        // Set up Data
        //int numRows = 10;
        //int numCols = 5;

        // Set up VectorizedPipeline (Fused)
        size_t numOutputs = 1;
        size_t numInputs = 1;

        DT** outputs = new DT*[numOutputs];
        outputs[0] = nullptr;;
        
        bool* isScalar = new bool[numInputs];
        isScalar[0] = false;

        Structure** inputs = new Structure*[numInputs];
        inputs[0] = m1;

        int64_t outRows[1] = {numCols}; // <- flipped
        int64_t outCols[1] = {numRows}; // <- flipped
        int64_t splits[1] = {1};
        int64_t combines[1] = {2};

        size_t numFuncs = 1;
        void (*funPtrs[1])(DT ***, Structure **, DaphneContext*) = {
            fusedVectorizedPipeline
        };
        void **fun = reinterpret_cast<void**>(funPtrs);

        meter.measure([&outputs, &numOutputs, &isScalar, &inputs, &numInputs, &outRows, &outCols, &splits, &combines, &numFuncs, &fun, &dctx]() {
            vectorizedPipeline<DT>(outputs, numOutputs, isScalar, inputs, numInputs, outRows, outCols, splits, combines, numFuncs, fun, dctx.get()); 
            return;
        });

        DataObjectFactory::destroy(outputs[0]);
    };
    DataObjectFactory::destroy(m1);
}