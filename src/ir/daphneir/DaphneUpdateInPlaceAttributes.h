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

#ifndef SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEATTRIBUTES_H
#define SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEATTRIBUTES_H

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace daphne {

enum class UpdateInPlaceAttrValue {
    NONE,
    LHS,
    RHS,
    BOTH,
 };

class UpdateInPlaceStorage : public mlir::AttributeStorage {
public:
    UpdateInPlaceStorage(UpdateInPlaceAttrValue type) : type(type) {}

    using KeyTy = UpdateInPlaceAttrValue;

    bool operator==(const KeyTy &key) const { return key == type; }

    static UpdateInPlaceStorage *construct(mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<UpdateInPlaceStorage>()) UpdateInPlaceStorage(key);
    }

    static llvm::hash_code hashKey(const KeyTy &key) { return llvm::hash_value(key); }

    UpdateInPlaceAttrValue type;
};

class UpdateInPlaceAttr : public mlir::Attribute::AttrBase<UpdateInPlaceAttr, mlir::Attribute,
        mlir::daphne::UpdateInPlaceStorage> {
public:
    using Base::Base;

    static UpdateInPlaceAttr get(mlir::MLIRContext *ctx, UpdateInPlaceAttrValue type) {
        return Base::get(ctx, type);
    }


    void print(llvm::raw_ostream &os) const {
        os << _key << "<" << this->getValueAsString() << ">";
    }

    StringRef getValueAsString() const {
        switch(getImpl()->type) {
            case UpdateInPlaceAttrValue::NONE:
                return "NONE";
            case UpdateInPlaceAttrValue::LHS:
                return "LHS";
            case UpdateInPlaceAttrValue::RHS:
                return "RHS";
            case UpdateInPlaceAttrValue::BOTH:
                return "BOTH";
        }
        return "UNKNOWN";
    }

    UpdateInPlaceAttrValue getValue() const { return getImpl()->type; }

    static StringRef getAttrName() { return _key; }

private:
    constexpr static const StringRef _key = "updateInPlace";

};

} // namespace daphne
} //namespace mlir

#endif // SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEATTRIBUTES_CPP

