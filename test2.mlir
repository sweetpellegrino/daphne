//IR after parsing:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %2 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %3 = "daphne.cast"(%1) : (si64) -> index
    %4 = "daphne.cast"(%2) : (si64) -> index
    %5 = "daphne.fill"(%0, %3, %4) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
    %6 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %7 = "daphne.ewAdd"(%5, %6) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
    %8 = "daphne.ewSqrt"(%7) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.return"() : () -> ()
  }
}
%0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.fill
%1 = "daphne.constant"() {value = 2 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.cast
%2 = "daphne.constant"() {value = 2 : si64} : () -> si64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.cast
%3 = "daphne.cast"(%1) : (si64) -> index
//Visiting op 'daphne.cast' with 1 operands:
//  - Operand produced by operation 'daphne.constant0x559eda0d2450'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.fill
%4 = "daphne.cast"(%2) : (si64) -> index
//Visiting op 'daphne.cast' with 1 operands:
//  - Operand produced by operation 'daphne.constant0x559eda0dbd30'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.fill
%5 = "daphne.fill"(%0, %3, %4) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.fill' with 3 operands:
//  - Operand produced by operation 'daphne.constant0x559eda0ca4c0'
//  - Operand produced by operation 'daphne.cast0x559eda0c9460'
//  - Operand produced by operation 'daphne.cast0x559eda0d6230'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.ewAdd
%6 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.ewAdd
%7 = "daphne.ewAdd"(%5, %6) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.ewAdd' with 2 operands:
//  - Operand produced by operation 'daphne.fill0x559eda0dbdf0'
//  - Operand produced by operation 'daphne.constant0x559eda0dbcd0'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.ewSqrt
%8 = "daphne.ewSqrt"(%7) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.ewSqrt' with 1 operands:
//  - Operand produced by operation 'daphne.ewAdd0x559eda0d0780'
//Has 1 results:
//  - Result 0 has no uses
"daphne.return"() : () -> ()
//Visiting op 'daphne.return' with 0 operands:
//Has 0 results:
func.func @main() {
  %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %1 = "daphne.constant"() {value = 2 : si64} : () -> si64
  %2 = "daphne.constant"() {value = 2 : si64} : () -> si64
  %3 = "daphne.cast"(%1) : (si64) -> index
  %4 = "daphne.cast"(%2) : (si64) -> index
  %5 = "daphne.fill"(%0, %3, %4) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
  %6 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %7 = "daphne.ewAdd"(%5, %6) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
  %8 = "daphne.ewSqrt"(%7) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
  "daphne.return"() : () -> ()
}
//Visiting op 'func.func' with 0 operands:
//Has 0 results:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %2 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %3 = "daphne.cast"(%1) : (si64) -> index
    %4 = "daphne.cast"(%2) : (si64) -> index
    %5 = "daphne.fill"(%0, %3, %4) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
    %6 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %7 = "daphne.ewAdd"(%5, %6) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
    %8 = "daphne.ewSqrt"(%7) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after parsing and some simplifications:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %2 = "daphne.fill"(%1, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
    %3 = "daphne.ewAdd"(%2, %1) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
    %4 = "daphne.ewSqrt"(%3) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.return"() : () -> ()
  }
}
//IR after SQL parsing:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %2 = "daphne.fill"(%1, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
    %3 = "daphne.ewAdd"(%2, %1) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
    %4 = "daphne.ewSqrt"(%3) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.return"() : () -> ()
  }
}
//IR after inference:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %2 = "daphne.fill"(%1, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %3 = "daphne.ewAdd"(%2, %1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    %4 = "daphne.ewSqrt"(%3) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.return"() : () -> ()
  }
}
//IR after type adaptation:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %2 = "daphne.fill"(%1, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %3 = "daphne.ewAdd"(%2, %1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    %4 = "daphne.ewSqrt"(%3) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.return"() : () -> ()
  }
}
//IR after vectorization
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %2 = "daphne.fill"(%1, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %3 = "daphne.ewAdd"(%2, %1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    %4 = "daphne.ewSqrt"(%3) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.return"() : () -> ()
  }
}
//IR after managing object references
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = 140721746457344 : ui64} : () -> ui64
    %3 = "daphne.createDaphneContext"(%2) : (ui64) -> !daphne.DaphneContext
    %4 = "daphne.fill"(%0, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %5 = "daphne.ewAdd"(%4, %0) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    "daphne.decRef"(%4) : (!daphne.Matrix<2x2xf64>) -> ()
    %6 = "daphne.ewSqrt"(%5) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.decRef"(%6) : (!daphne.Matrix<2x2xf64>) -> ()
    "daphne.decRef"(%5) : (!daphne.Matrix<2x2xf64>) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after kernel lowering
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = 140721746457344 : ui64} : () -> ui64
    %3 = "daphne.call_kernel"(%2) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
    %4 = "daphne.call_kernel"(%0, %1, %1, %3) {callee = "_fill__DenseMatrix_double__double__size_t__size_t"} : (f64, index, index, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64>
    %5 = "daphne.call_kernel"(%4, %0, %3) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__double"} : (!daphne.Matrix<2x2xf64>, f64, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64>
    "daphne.call_kernel"(%4, %3) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64>, !daphne.DaphneContext) -> ()
    %6 = "daphne.call_kernel"(%5, %3) {callee = "_ewSqrt__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x2xf64>, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64>
    "daphne.call_kernel"(%6, %3) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%5, %3) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%3) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after llvm lowering
module {
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_ewSqrt__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>)
  llvm.func @_fill__DenseMatrix_double__double__size_t__size_t(!llvm.ptr<ptr<i1>>, f64, i64, i64, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(140721746457344 : ui64) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %5 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %5, %4 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%4, %2) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %6 = llvm.load %4 : !llvm.ptr<ptr<i1>>
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %9 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %9, %8 : !llvm.ptr<ptr<i1>>
    llvm.call @_fill__DenseMatrix_double__double__size_t__size_t(%8, %0, %1, %1, %6) : (!llvm.ptr<ptr<i1>>, f64, i64, i64, !llvm.ptr<i1>) -> ()
    %10 = llvm.load %8 : !llvm.ptr<ptr<i1>>
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.alloca %11 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %13 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %13, %12 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__double(%12, %10, %0, %6) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>) -> ()
    %14 = llvm.load %12 : !llvm.ptr<ptr<i1>>
    %15 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%10, %6) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %18 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %18, %17 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewSqrt__DenseMatrix_double__DenseMatrix_double(%17, %14, %6) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %19 = llvm.load %17 : !llvm.ptr<ptr<i1>>
    %20 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%19, %6) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %21 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%14, %6) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %22 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%6) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}
