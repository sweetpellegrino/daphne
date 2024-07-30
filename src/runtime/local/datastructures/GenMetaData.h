/*
 * Copyright 2022 The DAPHNE Consortium
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

#pragma once

#include <runtime/local/datastructures/Structure.h>

#include "Structure.h"

template<typename ValueType>
class GenMetaData : public Structure {
public:
   GenMetaData(size_t numRows, size_t numCols) : Structure(numRows, numCols) {}

   using VT = ValueType;

   size_t getNumDims() const override {
       return 2;
   }

   size_t getNumItems() const override {
       return numRows * numCols;
   }

   void print(std::ostream & os) const override {
       throw std::runtime_error("Not implemented");
   }

   Structure* sliceRow(size_t rl, size_t ru) const override {
       throw std::runtime_error("Not implemented");
   }

   Structure* sliceCol(size_t cl, size_t cu) const override {
       throw std::runtime_error("Not implemented");
   }

   Structure* slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
       throw std::runtime_error("Not implemented");
   }

   size_t serialize(std::vector<char> &buf) const override {
       throw std::runtime_error("Not implemented");
   }
};