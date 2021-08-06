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

#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/kernels/Read.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEST_CASE("Read - Frame", TAG_KERNELS) {
    Frame * f = nullptr;
    read(f, "./test/runtime/local/io/ReadCsv4.csv", nullptr);
    
    CHECK(f->getNumRows() == 2);
    CHECK(f->getNumCols() == 2);
    CHECK(f->getColumnType(0) == ValueTypeCode::SI64);
    CHECK(f->getColumnType(1) == ValueTypeCode::F64);
    CHECK(f->getLabels()[0] == "foo");
    CHECK(f->getLabels()[1] == "bar");
    
    auto c0 = f->getColumn<int64_t>(0);
    CHECK(c0->get(0, 0) == 1);
    CHECK(c0->get(1, 0) == 2);
    auto c1 = f->getColumn<double>(1);
    CHECK(c1->get(0, 0) == 0.5);
    CHECK(c1->get(1, 0) == 1.0);
    
    DataObjectFactory::destroy(f);
    // TODO We cannot do this at the moment due to a bug regarding data sharing.
//    DataObjectFactory::destroy(c0);
//    DataObjectFactory::destroy(c1);
}