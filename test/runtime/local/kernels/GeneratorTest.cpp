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

#include "runtime/local/datastructures/GenMetaData.h"
#include "runtime/local/datastructures/Structure.h"
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/Generator.h>

#include <tags.h>

#include <catch.hpp>
#include <string>
#include <type_traits>
#include <vector>
#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES double, int64_t, uint32_t

TEMPLATE_PRODUCT_TEST_CASE("TEST generator", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    
    GenMetaData<int>* res = nullptr;
    size_t numRows = 10;
    size_t numCols = 5;

    generator(res, -1, numRows, numCols, nullptr);

    auto res_rows = res->getNumRows();
    auto res_cols = res->getNumCols();

    CHECK(res_rows == numRows);
    CHECK(res_cols == numCols);

    DataObjectFactory::destroy(res);
}