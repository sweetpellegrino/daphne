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

/*template<class DTRes, typename DTArg>
struct FusedPipelineStruct {
    static void fusedVectorizedPipeline(DTRes *&res, DTArg *row, DaphneContext *ctx) {
        std::cout << "Test123" << std::endl;
        return ewUnaryMat<DTRes, DTArg>(UnaryOpCode::SQRT, res, row, ctx);
    }
};

static void fusedVectorizedPipeline(DenseMatrix<double> *&res, DenseMatrix<double> *row, DaphneContext *ctx) {
        std::cout << "Test123" << std::endl;
        ewUnaryMat<DenseMatrix<double>, DenseMatrix<double>>(UnaryOpCode::SQRT, res, row, ctx);
}*/

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
static void fusedVectorizedPipeline(DenseMatrix<double> ***res, Structure **row, DaphneContext *ctx) {
    DenseMatrix<double> *& _res = *res[1];
    DenseMatrix<double> *_row = reinterpret_cast<DenseMatrix<double>*>(row[1]);
    std::cout << "Test" << std::endl;
    ewUnaryMat<DenseMatrix<double>, DenseMatrix<double>>(UnaryOpCode::SQRT, _res, _row, ctx);
}

typedef struct
{
   void (*ptr)(void);
} Func;

template<class DT,typename VT>
void generateBinaryMatrices(DT *& m1, DT *& m2, size_t numRows, size_t numCols, VT val1, VT val2) {
    fillMatrix<DT, VT>(m1, numRows, numCols, val1);
    fillMatrix<DT, VT>(m2, numRows, numCols, val2);
}

TEMPLATE_PRODUCT_TEST_CASE("VectorizedPipeline - Fused Ops", TAG_VECTORIZED_BENCH, (DenseMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    BENCHMARK_ADVANCED("MAIN") (Catch::Benchmark::Chronometer meter) {
        
        auto dctx = setupContextAndLogger();

        /* Set up Data */

        int numRows = 1000;
        int numCols = 1000;

        DT* m1 = nullptr;
        DT* m2 = nullptr;
        generateBinaryMatrices<DT, VT>(m1, m2, numRows, numCols, VT(1), VT(2));
        DT* res = nullptr;

        /* Set up VectorizedPipeline-Kernel */

        size_t numOutputs = 1;
        size_t numInputs = 1;

        DT** outputs = new DT*[numOutputs];
        outputs[0] = res;
        
        bool* isScalar = new bool[numInputs];
        isScalar[0] = false;

        Structure** inputs = new Structure*[numInputs];
        inputs[0] = m1;

        int64_t outRows[1] = {numRows};
        int64_t outCols[1] = {numCols};
        int64_t splits[1] = {1};
        int64_t combines[1] = {1};

        size_t numFuncs = 1;
        void (*funPtrs[1])(DT ***, Structure **, DaphneContext*) = {
            fusedVectorizedPipeline
        };
        void **fun = reinterpret_cast<void**>(funPtrs);

        vectorizedPipeline<DT>(outputs, numOutputs, isScalar, inputs, numInputs, outRows, outCols, splits, combines, numFuncs, fun, dctx.get()); 

        /*std::function<void(DT *&res, DT *row, DaphneContext *ctx)> func = [&] (DT*& res, DT* row, DaphneContext* ctx) {
            ewUnaryMat<DT, DT>(UnaryOpCode::SQRT, res, row, nullptr);
        }tFused.fusedVectorizedPipeline;
        }

        /*auto fusedVectorizedPipeline = [](DT *&res, DT *row, DaphneContext *ctx) {
            std::cout << "Test123" << std::endl;
            return ewUnaryMat<DT, DT>(UnaryOpCode::SQRT, res, row, ctx);
        };*/

        //std::function<void(DT*&, DT*, DaphneContext*)> *func = new std::function<void(DT*&, DT*, DaphneContext*)>(fusedVectorizedPipeline);

        //vectorizedPipeline<DT>(outputs, numOutputs, isScalar, inputs, numInputs, outRows, outCols, splits, combines, numFuncs, fun, dctx.get()); 

       /* meter.measure([&m1, &m2, &res]() {

            return vectorizedPipeline<DT>(outputs, numOutputs, isScalar, inputs, numInputs, outRows, outCols, splits, combines numFuncs, fun, nullptr); 
        });*/

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(m2);
        DataObjectFactory::destroy(res);
    };
}

/*TEMPLATE_PRODUCT_TEST_CASE("VectorizedPipeline - Sequential Pipes", TAG_VECTORIZED_BENCH, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    using VT = typename DT::VT;

    BENCHMARK_ADVANCED("MAIN") (Catch::Benchmark::Chronometer meter) {
        DT* m1 = nullptr;
        DT* m2 = nullptr;
        generateBinaryMatrices<DT, VT>(m1, m2, 5000, 5000, VT(1), VT(2));
        generateBinaryMatrices<DT, VT>(m1, m2, 5000, 5000, VT(1), VT(2));
        DT* res = nullptr;

        meter.measure([&m1, &m2, &res]() {
            return ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, res, m1, m2, nullptr);
        });

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(m2);
        DataObjectFactory::destroy(res);
    };
}*/