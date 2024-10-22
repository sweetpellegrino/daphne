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

#include "runtime/local/vectorized/Tasks.h"
#include "ir/daphneir/Daphne.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/kernels/BinaryOpCode.h"
#include "runtime/local/kernels/EwBinaryMat.h"
#include <cstdint>
#include <cstdio>
#include <llvm/Support/raw_ostream.h>
#include <stdexcept>
#include <chrono>

#ifdef USE_PAPI
#include <papi.h>
#endif

template <typename VT> void CompiledPipelineTask<DenseMatrix<VT>>::execute(uint32_t fid, uint32_t batchSize) {
    // local add aggregation to minimize locking
    std::vector<DenseMatrix<VT> *> localAddRes(_data._numOutputs);
    std::vector<DenseMatrix<VT> *> localResults(_data._numOutputs);
    std::vector<DenseMatrix<VT> **> outputs;
    for (auto &lres : localResults)
        outputs.push_back(&lres);
    for (uint64_t d = _data._dl; d < _data._du; d += batchSize) {
        // create zero-copy views of inputs/outputs
        uint64_t d2 = std::min(d + batchSize, _data._du);

        auto linputs = this->createFuncInputs(d, d2);

        // execute function on given data binding (batch size)
        _data._funcs[fid](outputs.data(), linputs.data(), _data._ctx);
        accumulateOutputs(localResults, localAddRes, d, d2);

        // cleanup
        for (auto &localResult : localResults)
            if (localResult) {
                DataObjectFactory::destroy(localResult);
                localResult = nullptr;
            }

        // Note that a pipeline manages the reference counters of its inputs
        // internally. Thus, we do not need to care about freeing the inputs
        // here.
    }

    for (size_t o = 0; o < _data._numOutputs; ++o) {

        if (_data._combines[o] == VectorCombine::ROWS || _data._combines[o] == VectorCombine::COLS)
            continue;

        auto &result = (*_res[o]);
        _resLock.lock();
        if (result == nullptr) {
            result = localAddRes[o];
            _resLock.unlock();
        } else {
            switch (_data._combines[o]) {
                case VectorCombine::ADD:
                    ewBinaryMat(BinaryOpCode::ADD, result, result, localAddRes[o], _data._ctx);
                    break;
                case VectorCombine::MIN:
                    ewBinaryMat(BinaryOpCode::MIN, result, result, localAddRes[o], _data._ctx);
                    break;
                case VectorCombine::MAX:
                    ewBinaryMat(BinaryOpCode::MAX, result, result, localAddRes[o], _data._ctx);
                    break;
                default:
                    throw std::runtime_error("not implemented");
                    break;
            }
            _resLock.unlock();
            // cleanup
            DataObjectFactory::destroy(localAddRes[o]);
        }
    }
}

template <typename VT> uint64_t CompiledPipelineTask<DenseMatrix<VT>>::getTaskSize() { return _data._du - _data._dl; }

template <typename VT>
void CompiledPipelineTask<DenseMatrix<VT>>::accumulateOutputs(std::vector<DenseMatrix<VT> *> &localResults,
                                                              std::vector<DenseMatrix<VT> *> &localAddRes,
                                                              uint64_t dimStart, uint64_t dimEnd) {
    // TODO: in-place computation via better compiled pipelines
    // TODO: multi-return
    for (auto o = 0u; o < _data._numOutputs; ++o) {
        auto &result = (*_res[o]);
        switch (_data._combines[o]) {
            case VectorCombine::ROWS: {
                auto slice = result->sliceRow(dimStart - _data._offset, dimEnd - _data._offset);
                accumulateConcat(slice, localResults[o]);

                DataObjectFactory::destroy(slice);
                break;
            }
            case VectorCombine::COLS: {
                auto slice = result->sliceCol(dimStart - _data._offset, dimEnd - _data._offset);
                accumulateConcat(slice, localResults[o]);

                DataObjectFactory::destroy(slice);
                break;
            }
            case VectorCombine::ADD: {
                accumulateAggregate(localAddRes[o], localResults[0], BinaryOpCode::ADD);
                break;
            }
            case VectorCombine::MAX: {
                accumulateAggregate(localAddRes[o], localResults[0], BinaryOpCode::MAX);
                break;
            }
            case VectorCombine::MIN: {
                accumulateAggregate(localAddRes[o], localResults[0], BinaryOpCode::MIN);
                break;
            }
            default: {
                throw std::runtime_error(("VectorCombine case `" +
                                        std::to_string(static_cast<int64_t>(_data._combines[o])) + "` not supported"));
            }
        }
    }
}

template<typename VT>
void CompiledPipelineTask<DenseMatrix<VT>>::accumulateAggregate(DenseMatrix<VT>*& localAddRes,
                                                                DenseMatrix<VT>*& localResult,
                                                                BinaryOpCode opCode) {
    if (localAddRes == nullptr) {
        // take lres and reset it to nullptr
        localAddRes = localResult;
        localResult = nullptr;
    } else {
        ewBinaryMat(opCode, localAddRes, localAddRes, localResult, nullptr);
    }
}

template<typename VT>
void CompiledPipelineTask<DenseMatrix<VT>>::accumulateConcat(DenseMatrix<VT>*& slice,
                                                             DenseMatrix<VT>*& localResult) {

    /*llvm::outs() << "+++++++++++++++++++++++++++++++++++++" << "\n";

    llvm::outs() << "s numRows: " << slice->getNumRows() << "\n";
    llvm::outs() << "s numCols: " << slice->getNumCols() << "\n";
    llvm::outs() << "s rowSkip: " << slice->getRowSkip() << "\n";
    llvm::outs() << "s isRowMajor: " << slice->getIsRowMajor() << "\n";

    llvm::outs() << "dimStart: " << dimStart << "\n";
    llvm::outs() << "dimEnd: " << dimEnd << "\n";
    llvm::outs() << "offset: "   << _data._offset << "\n";

    llvm::outs() << "lo numRows: "   << localResults[o]->getNumRows()  << "\n";
    llvm::outs() << "lo numCols: "   << localResults[o]->getNumCols()  << "\n";
    llvm::outs() << "lo rowSkip: "   << localResults[o]->getRowSkip()  << "\n";
    llvm::outs() << "lo isRowMajor: "   << localResults[o]->getIsRowMajor()  << "\n";

    llvm::outs() << "r numRows: "   << result->getNumRows()  << "\n";
    llvm::outs() << "r numCols: "   << result->getNumCols()  << "\n";
    llvm::outs() << "r rowSkip: "   << result->getRowSkip()  << "\n";
    llvm::outs() << "r isRowMajor: "   << result->getIsRowMajor()  << "\n";*/

    if (slice->getIsRowMajor() != localResult->getIsRowMajor())
        llvm::outs() << "?????????" << "\n"; 

    //PAPI_hl_region_begin("fixme_cols");
    auto start = std::chrono::high_resolution_clock::now();
    VT *sliceValues = slice->getValues();
    VT *localResultsValues = localResult->getValues();
           
    if (slice->getIsRowMajor()) {
        for (auto i = 0u; i < slice->getNumRows(); ++i) {
            for (auto j = 0u; j < slice->getNumCols(); ++j) {
                sliceValues[i * slice->getRowSkip() + j] =
                    localResultsValues[i * localResult->getRowSkip() + j];
            }
        }
    }
    else {
        for (auto j = 0u; j < slice->getNumCols(); ++j) {
            for (auto i = 0u; i < slice->getNumRows(); ++i) {
                sliceValues[j * slice->getRowSkip() + i] =
                    localResultsValues[j * localResult->getRowSkip() + i];
            }
        }
    }
    //PAPI_hl_region_end("fixme_rows");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    llvm::outs() << "isRowMajor: " << slice->getIsRowMajor() << ": " << elapsed.count() << " ms\n";

    /*for (size_t i = 0; i < slice->getNumItems(); ++i) {
        llvm::outs() << "i(" << i << "): " << slice->getValues()[i] << "\n";
    }
    llvm::outs() << "\n";

    llvm::outs() << "result" << "\n"; 
    for (size_t i = 0; i < result->getNumItems(); ++i) {
        llvm::outs() << "i(" << i << "): " << result->getValues()[i] << "\n";
    }
    llvm::outs() << "\n";*/

}

//-----------------------------------------------------------------------------

template <typename VT> void CompiledPipelineTask<CSRMatrix<VT>>::execute(uint32_t fid, uint32_t batchSize) {
    std::vector<size_t> localResNumRows(_data._numOutputs);
    std::vector<size_t> localResNumCols(_data._numOutputs);
    for (size_t i = 0; i < _data._numOutputs; i++) {
        switch (_data._combines[i]) {
        case VectorCombine::ROWS: {
            if (_data._wholeResultCols[i] == -1)
                throw std::runtime_error("TODO: CompiledPipeLineTask (CSRMatrix) Rows "
                                         "_data._wholeResultCols[i] == -1");
            localResNumRows[i] = _data._du - _data._dl;
            localResNumCols[i] = _data._wholeResultCols[i];
            break;
        }
        case VectorCombine::COLS: {
            if (_data._wholeResultRows[i] == -1)
                throw std::runtime_error("TODO: CompiledPipeLineTask (CSRMatrix) Cols "
                                         "_data._wholeResultRows[i] == -1");
            localResNumRows[i] = _data._wholeResultRows[i];
            localResNumCols[i] = _data._du - _data._dl;
            break;
        }
        default:
            throw std::runtime_error("Not implemented");
        }
    }

    std::vector<VectorizedDataSink<CSRMatrix<VT>> *> localSinks(_data._numOutputs);
    for (size_t i = 0; i < _data._numOutputs; i++)
        localSinks[i] =
            new VectorizedDataSink<CSRMatrix<VT>>(_data._combines[i], localResNumRows[i], localResNumCols[i]);

    std::vector<CSRMatrix<VT> *> lres(_data._numOutputs, nullptr);
    for (uint64_t d = _data._dl; d < _data._du; d += batchSize) {
        // create zero-copy views of inputs/outputs
        uint64_t d2 = std::min(d + batchSize, _data._du);

        auto linputs = this->createFuncInputs(d, d2);
        CSRMatrix<VT> ***outputs = new CSRMatrix<VT> **[_data._numOutputs];
        for (size_t i = 0; i < _data._numOutputs; i++)
            outputs[i] = &(lres[i]);
        // execute function on given data binding (batch size)
        _data._funcs[fid](outputs, linputs.data(), _data._ctx);
        delete[] outputs;
        for (size_t i = 0; i < _data._numOutputs; i++)
            localSinks[i]->add(lres[i], d - _data._dl, false);

        // cleanup
        for (size_t i = 0; i < _data._numOutputs; i++)
            lres[i] = nullptr;

        // Note that a pipeline manages the reference counters of its inputs
        // internally. Thus, we do not need to care about freeing the inputs
        // here.
    }
    for (size_t i = 0; i < _data._numOutputs; i++) {
        _resultSinks[i]->add(localSinks[i]->consume(), _data._dl);
        delete localSinks[i];
    }
}

template <typename VT> uint64_t CompiledPipelineTask<CSRMatrix<VT>>::getTaskSize() { return _data._du - _data._dl; }

template class CompiledPipelineTask<DenseMatrix<double>>;
template class CompiledPipelineTask<DenseMatrix<float>>;
template class CompiledPipelineTask<DenseMatrix<int64_t>>;

template class CompiledPipelineTask<CSRMatrix<double>>;
template class CompiledPipelineTask<CSRMatrix<float>>;
