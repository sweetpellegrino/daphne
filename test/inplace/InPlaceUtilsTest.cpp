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

#include <catch.hpp>

#include <runtime/local/kernels/InPlaceUtils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <tags.h>

//
TEST_CASE("inPlaceAdd", TAG_INPLACE) {
    
    //Create DenseMatrix
    const size_t numRows = 10000;
    const size_t numCols = 2000;
    
    DenseMatrix<double> * m = DataObjectFactory::create<DenseMatrix<double>>(numRows, numCols, false);

}