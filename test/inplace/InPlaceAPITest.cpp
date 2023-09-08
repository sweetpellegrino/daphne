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

#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

const std::string dirPath = "test/inplace/daphne/";

#define MAKE_TEST_CASE(name, suffix, param) \
    TEST_CASE(std::string(name)+std::string(suffix), TAG_INPLACE) { \
        std::string prefix(dirPath);\
        prefix += (name);\
        compareDaphneToRef(prefix + ".txt", prefix + ".daphne", (param)); \
    }

MAKE_TEST_CASE("bfu-matrix-test", "", "--update-in-place")
MAKE_TEST_CASE("lfu-matrix-test", "", "--update-in-place")
MAKE_TEST_CASE("rfu-matrix-test", "", "--update-in-place")
MAKE_TEST_CASE("nfu-matrix-test", "", "--update-in-place")

//ToDo: make these tests work
//#ifdef USE_CUDA
//MAKE_TEST_CASE("runMatMult", "CUDA", "--vec --cuda")
//MAKE_TEST_CASE("runEwBinary", "CUDA", "--vec --cuda")
//MAKE_TEST_CASE("runRowAgg", "CUDA", "--vec --cuda")
//MAKE_TEST_CASE("runColAgg", "CUDA", "--vec --cuda")
//MAKE_TEST_CASE("runOther", "CUDA", "--vec --cuda")
//#endif
