/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLEOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLEOPINTERFACE_H
namespace mlir::OpTrait {
    template<class ConcreteOp>
    class CUDASupport : public TraitBase<ConcreteOp, CUDASupport> {};

    template<class ConcreteOp>
    class VectorElementWise : public TraitBase<ConcreteOp, VectorElementWise> {};

    /*template<size_t i>
    struct VectorTrait {
        template<class ConcreteOp>
        class Impl: public TraitBase<ConcreteOp, Impl> {};
    };

    template<class ConcreteOp>
    class VectorElementWise : public TraitBase<ConcreteOp, VectorElementWise> {};*/

    template<class ConcreteOp>
    class VectorReduction : public TraitBase<ConcreteOp, VectorReduction> {};

    template<class ConcreteOp>
    class VectorTranspose : public TraitBase<ConcreteOp, VectorTranspose> {};

    template<class ConcreteOp>
    class VectorMatMul : public TraitBase<ConcreteOp, VectorMatMul> {};
}
namespace mlir::daphne {
#include <ir/daphneir/DaphneVectorizableOpInterface.h.inc>
}

#endif // SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLEOPINTERFACE_H