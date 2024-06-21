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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_AGGROW_H
#define SRC_RUNTIME_LOCAL_KERNELS_AGGROW_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/CSCMatrix.h>
#include <runtime/local/datastructures/MCSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <vector>

#include <cstddef>
#include <cstring>
#include <cmath>
#include <typeinfo>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct AggRow {
    static void apply(AggOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void aggRow(AggOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    AggRow<DTRes, DTArg>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct AggRow<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);

        const VTArg * valuesArg = arg->getValues();
        VTRes * valuesRes = res->getValues();

        if(opCode == AggOpCode::IDXMIN) {
            for(size_t r = 0; r < numRows; r++) {
                VTArg minVal = valuesArg[0];
                size_t minValIdx = 0;
                for(size_t c = 1; c < numCols; c++)
                    if(valuesArg[c] < minVal) {
                        minVal = valuesArg[c];
                        minValIdx = c;
                    }
                *valuesRes = static_cast<VTRes>(minValIdx);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        }
        else if(opCode == AggOpCode::IDXMAX) {
            for(size_t r = 0; r < numRows; r++) {
                VTArg maxVal = valuesArg[0];
                size_t maxValIdx = 0;
                for(size_t c = 1; c < numCols; c++)
                    if(valuesArg[c] > maxVal) {
                        maxVal = valuesArg[c];
                        maxValIdx = c;
                    }
                *valuesRes = static_cast<VTRes>(maxValIdx);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        }
        else {
            EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;
            if(AggOpCodeUtils::isPureBinaryReduction(opCode))
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
            else
                // TODO Setting the function pointer yields the correct result.
                // However, since MEAN and STDDEV are not sparse-safe, the program
                // does not take the same path for doing the summation, and is less
                // efficient.
                // for MEAN and STDDDEV, we need to sum
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

            for(size_t r = 0; r < numRows; r++) {
                VTRes agg = static_cast<VTRes>(*valuesArg);
                for(size_t c = 1; c < numCols; c++){
                    agg = func(agg, static_cast<VTRes>(valuesArg[c]), ctx);
                }
                *valuesRes = static_cast<VTRes>(agg);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }

            if(AggOpCodeUtils::isPureBinaryReduction(opCode))
                return;

            // The op-code is either MEAN or STDDEV or VAR
            valuesRes = res->getValues();
            // valuesArg = arg->getValues();
            for(size_t r = 0; r < numRows; r++) {
                *valuesRes = (*valuesRes) / numCols;
                valuesRes += res->getRowSkip();
            }

            if(opCode == AggOpCode::MEAN)
                return;
            
            // else op-code is STDDEV or VAR

            // Create a temporary matrix to store the resulting standard deviations for each row
            auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, true);
            VTRes * valuesT = tmp->getValues();
            valuesArg = arg->getValues();
            valuesRes = res->getValues();
            for(size_t r = 0; r < numRows; r++) {
                for(size_t c = 0; c < numCols; c++) {
                    VTRes val = static_cast<VTRes>(valuesArg[c]) - (*valuesRes);
                    valuesT[r] = valuesT[r] + val * val;
                }
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
               
            }
            valuesRes = res->getValues();
            for(size_t c = 0; c < numRows; c++) {
                valuesT[c] /= numCols;
                if(opCode == AggOpCode::STDDEV)
                    *valuesRes = sqrt(valuesT[c]);
                else
                    *valuesRes = valuesT[c];
                valuesRes += res->getRowSkip();
            }

            DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);
            
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct AggRow<DenseMatrix<VTRes>, CSRMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const CSRMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);

        VTRes * valuesRes = res->getValues();

        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {

            EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));

            const bool isSparseSafe = AggOpCodeUtils::isSparseSafe(opCode);
            const VTRes neutral = AggOpCodeUtils::template getNeutral<VTRes>(opCode);

            for(size_t r = 0; r < numRows; r++) {
                *valuesRes = AggAll<VTRes, CSRMatrix<VTArg>>::aggArray(
                        arg->getValues(r),
                        arg->getNumNonZeros(r),
                        numCols,
                        func,
                        isSparseSafe,
                        neutral,
                        ctx
                );
                valuesRes += res->getRowSkip();
            }
        }
        else { // The op-code is either MEAN or STDDEV or VAR
            // get sum for each row
            size_t ctr = 0 ;
            const VTRes neutral = VTRes(0);
            const bool isSparseSafe = true;
            auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, true);
            VTRes * valuesT = tmp->getValues();
            EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));
            for (size_t r = 0; r < numRows; r++){
                *valuesRes = AggAll<VTRes, CSRMatrix<VTArg>>::aggArray(
                    arg->getValues(r),
                    arg->getNumNonZeros(r),
                    numCols,
                    func,
                    isSparseSafe,
                    neutral,
                    ctx
                );
                const VTArg * valuesArg = arg->getValues(0);
                const size_t numNonZeros = arg->getNumNonZeros(r);
                *valuesRes = *valuesRes / numCols;
                if (opCode != AggOpCode::MEAN){
                    for(size_t i = ctr; i < ctr+numNonZeros; i++) {
                        VTRes val = static_cast<VTRes>((valuesArg[i])) - (*valuesRes);
                        valuesT[r] = valuesT[r] + val * val;
                    }

                    ctr+=numNonZeros; 
                    valuesT[r] += (numCols - numNonZeros) * (*valuesRes)*(*valuesRes);
                    valuesT[r] /= numCols;
                    if(opCode == AggOpCode::STDDEV)
                        *valuesRes = sqrt(valuesT[r]);
                    else
                        *valuesRes = valuesT[r];
                }
                valuesRes += res->getRowSkip();
            }
            valuesRes = res->getValues();
            DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);

        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct AggRow<Matrix<VTRes>, Matrix<VTArg>> {
    static void apply(AggOpCode opCode, Matrix<VTRes> *& res, const Matrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);
        
        if (opCode == AggOpCode::IDXMIN) {
            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r) {
                VTArg minVal = arg->get(r, 0);
                size_t minValIdx = 0;
                for (size_t c = 1; c < numCols; ++c) {
                    VTArg argVal = arg->get(r, c);
                    if (argVal < minVal) {
                        minVal = argVal;
                        minValIdx = c;
                    }
                }
                res->append(r, 0, static_cast<VTRes>(minValIdx));
            }
            res->finishAppend();
        }
        else if (opCode == AggOpCode::IDXMAX) {
            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r) {
                VTArg maxVal = arg->get(r, 0);
                size_t maxValIdx = 0;
                for (size_t c = 1; c < numCols; ++c) {
                    VTArg argVal = arg->get(r, c);
                    if (argVal > maxVal) {
                        maxVal = argVal;
                        maxValIdx = c;
                    }
                }
                res->append(r, 0, static_cast<VTRes>(maxValIdx));
            }
            res->finishAppend();
        }
        else {
            EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;    
            if (AggOpCodeUtils::isPureBinaryReduction(opCode))
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
            else
                // TODO Setting the function pointer yields the correct result.
                // However, since MEAN and STDDEV are not sparse-safe, the program
                // does not take the same path for doing the summation, and is less
                // efficient.
                // for MEAN and STDDDEV, we need to sum
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r) {
                VTRes agg = static_cast<VTRes>(arg->get(r, 0));
                for (size_t c = 1; c < numCols; ++c)
                    agg = func(agg, static_cast<VTRes>(arg->get(r, c)), ctx);
                res->append(r, 0, static_cast<VTRes>(agg));
            }
            res->finishAppend();

            if (AggOpCodeUtils::isPureBinaryReduction(opCode))
                return;

            // The op-code is either MEAN or STDDEV or VAR
            for (size_t r = 0; r < numRows; ++r) {
                res->set(r, 0, res->get(r, 0) / numCols);
            }

            if (opCode == AggOpCode::MEAN)
                return;
            
            // else op-code is STDDEV or VAR

            // Create a temporary matrix to store the resulting standard deviations for each row
            std::vector<VTRes> tmp(numRows);

            for (size_t r = 0; r < numRows; ++r) {
                for (size_t c = 0; c < numCols; ++c) {
                    VTRes val = static_cast<VTRes>(arg->get(r, c)) - res->get(r, 0);
                    tmp[r] += val * val;
                }               
            }

            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r) {
                tmp[r] /= numCols;
                if (opCode == AggOpCode::STDDEV)
                    res->append(r, 0, sqrt(tmp[r]));
                else
                    res->append(r, 0, tmp[r]);
            }
            res->finishAppend();
        }
    }
};


// ----------------------------------------------------------------------------
// DenseMatrix <- MCSRMatrix
// ----------------------------------------------------------------------------


template<typename VTRes, typename VTArg>
struct AggRow<DenseMatrix<VTRes>, MCSRMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const MCSRMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);

        VTRes * valuesRes = res->getValues();

        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
            EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
            const VTRes neutral = AggOpCodeUtils::template getNeutral<VTRes>(opCode);
            const bool isSparseSafe = AggOpCodeUtils::isSparseSafe(opCode);

            for(size_t r = 0; r < numRows; r++) {
                const VTArg* rowValues = arg->getValues(r);
                //const size_t* colIdxs = arg->getColIdxs(r);
                size_t nnz = arg->getNumNonZeros(r);
                VTRes aggResult = neutral;

                for(size_t idx = 0; idx < nnz; idx++) {
                    aggResult = func(aggResult, rowValues[idx], ctx);
                }

                if (isSparseSafe) {
                    size_t totalZeros = arg->getNumCols() - nnz;
                    for(size_t z = 0; z < totalZeros; z++) {
                        aggResult = func(aggResult, static_cast<VTArg>(0), ctx);
                    }
                }

                *valuesRes = aggResult;
                valuesRes += res->getRowSkip();
            }
        }
        else { // Handle cases like MEAN, STDDEV, etc.
            switch (opCode) {
                case AggOpCode::MEAN: {
                    for (size_t r = 0; r < numRows; r++) {
                        const VTArg* rowValues = arg->getValues(r);
                        size_t nnz = arg->getNumNonZeros(r);
                        VTRes sum = 0;

                        for(size_t idx = 0; idx < nnz; idx++) {
                            sum += rowValues[idx];
                        }

                        *valuesRes = sum / arg->getNumCols();
                        valuesRes += res->getRowSkip();
                    }
                    break;
                }
                // TODO: Implement for other aggregation operations like STDDEV, etc.
                default:
                    throw std::runtime_error("AggRow(MCSR) - Unsupported AggOpCode");
            }
        }
    }
};




// ----------------------------------------------------------------------------
// DenseMatrix <- CSCMatrix
// ----------------------------------------------------------------------------



template<typename VTRes, typename VTArg>
struct AggRow<DenseMatrix<VTRes>, CSCMatrix<VTArg>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *& res, const CSCMatrix<VTArg> * arg, DCTX(ctx)) {
       const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, true);

        VTRes * valuesRes = res->getValues();

        EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;
        if(AggOpCodeUtils::isPureBinaryReduction(opCode))
            func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
        else
            // TODO Setting the function pointer yields the correct result.
            // However, since MEAN and STDDEV are not sparse-safe, the program
            // does not take the same path for doing the summation, and is less
            // efficient.
            // for MEAN and STDDDEV, we need to sum
            func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

        const VTArg * valuesArg = arg->getValues(0);
        const size_t * rowIdxsArg = arg->getRowIdxs(0);

        const size_t numNonZeros = arg->getNumNonZeros();

        if(AggOpCodeUtils::isSparseSafe(opCode)) {
            for(size_t i = 0; i < numNonZeros; i++) {
                const size_t rowIdx = rowIdxsArg[i];
                valuesRes[rowIdx] = func(valuesRes[rowIdx], static_cast<VTRes>(valuesArg[i]), ctx);
            }
        }
        else {
            size_t * hist = new size_t[numCols](); // initialized to zeros

            const size_t numNonZerosFirstRowArg = arg->getNumNonZeros(0);
            for(size_t i = 0; i < numNonZerosFirstRowArg; i++) {
                size_t colIdx = rowIdxsArg[i];
                valuesRes[colIdx] = static_cast<VTRes>(valuesArg[i]);
                hist[colIdx]++;
            }

            if(arg->getNumRows() > 1) {
                for(size_t i = numNonZerosFirstRowArg; i < numNonZeros; i++) {
                    const size_t colIdx = rowIdxsArg[i];
                    valuesRes[colIdx] = func(valuesRes[colIdx], static_cast<VTRes>(valuesArg[i]), ctx);
                    hist[colIdx]++;
                }
                for(size_t c = 0; c < numRows; c++)
                    if(hist[c] < numCols)
                        valuesRes[c] = func(valuesRes[c], VTRes(0), ctx);
            }

            delete[] hist;
        }

        if(AggOpCodeUtils::isPureBinaryReduction(opCode))
            return;
        
        // The op-code is either MEAN or STDDEV or VAR.

        for(size_t c = 0; c < numCols; c++)
            valuesRes[c] /= arg->getNumRows();

        if(opCode == AggOpCode::MEAN)
            return;

        auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, true);
        VTRes * valuesT = tmp->getValues();

        size_t * nnzCol = new size_t[numCols](); // initialized to zeros
        for(size_t i = 0; i < numNonZeros; i++) {
            const size_t colIdx = rowIdxsArg[i];
            VTRes val = static_cast<VTRes>(valuesArg[i]) - valuesRes[colIdx];
            valuesT[colIdx] = valuesT[colIdx] + val * val;
            nnzCol[colIdx]++;
        }

        for(size_t c = 0; c < numCols; c++) {
            // Take all zeros in the column into account.
            valuesT[c] += (valuesRes[c] * valuesRes[c]) * (numRows - nnzCol[c]);
            // Finish computation of stddev.
            valuesT[c] /= numRows;
            if (opCode == AggOpCode::STDDEV)
                valuesT[c] = sqrt(valuesT[c]);
        }

        delete[] nnzCol;

        // TODO We could avoid copying by returning tmp and destroying res. But
        // that might be wrong if res was not nullptr initially.
        memcpy(valuesRes, valuesT, numCols * sizeof(VTRes));
        DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);

    }
};


#endif //SRC_RUNTIME_LOCAL_KERNELS_AGGROW_H
