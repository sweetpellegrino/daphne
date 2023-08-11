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
    %9 = "daphne.constant"() {value = true} : () -> i1
    %10 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%8, %9, %10) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
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
//  - Operand produced by operation 'daphne.constant0x5594cb4105f0'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.fill
%4 = "daphne.cast"(%2) : (si64) -> index
//Visiting op 'daphne.cast' with 1 operands:
//  - Operand produced by operation 'daphne.constant0x5594cb40fa50'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.fill
%5 = "daphne.fill"(%0, %3, %4) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.fill' with 3 operands:
//  - Operand produced by operation 'daphne.constant0x5594cb418f80'
//  - Operand produced by operation 'daphne.cast0x5594cb415dc0'
//  - Operand produced by operation 'daphne.cast0x5594cb422920'
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
//  - Operand produced by operation 'daphne.fill0x5594cb415880'
//  - Operand produced by operation 'daphne.constant0x5594cb4129e0'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.ewSqrt
%8 = "daphne.ewSqrt"(%7) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
//Visiting op 'daphne.ewSqrt' with 1 operands:
//  - Operand produced by operation 'daphne.ewAdd0x5594cb422de0'
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%9 = "daphne.constant"() {value = true} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
%10 = "daphne.constant"() {value = false} : () -> i1
//Visiting op 'daphne.constant' with 0 operands:
//Has 1 results:
//  - Result 0 has a single use: 
//    - daphne.print
"daphne.print"(%8, %9, %10) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
//Visiting op 'daphne.print' with 3 operands:
//  - Operand produced by operation 'daphne.ewSqrt0x5594cb422810'
//  - Operand produced by operation 'daphne.constant0x5594cb411e30'
//  - Operand produced by operation 'daphne.constant0x5594cb416f50'
//Has 0 results:
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
  %9 = "daphne.constant"() {value = true} : () -> i1
  %10 = "daphne.constant"() {value = false} : () -> i1
  "daphne.print"(%8, %9, %10) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
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
    %9 = "daphne.constant"() {value = true} : () -> i1
    %10 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%8, %9, %10) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//Visiting op 'builtin.module' with 0 operands:
//Has 0 results:
//IR after parsing and some simplifications:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
    %5 = "daphne.ewAdd"(%4, %3) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
    %6 = "daphne.ewSqrt"(%5) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%6, %2, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after SQL parsing:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
    %5 = "daphne.ewAdd"(%4, %3) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
    %6 = "daphne.ewSqrt"(%5) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%6, %2, %1) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after inference:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %5 = "daphne.ewAdd"(%4, %3) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    %6 = "daphne.ewSqrt"(%5) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.print"(%6, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after type adaptation:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %5 = "daphne.ewAdd"(%4, %3) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    %6 = "daphne.ewSqrt"(%5) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.print"(%6, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after vectorization
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %5 = "daphne.ewAdd"(%4, %3) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    %6 = "daphne.ewSqrt"(%5) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.print"(%6, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after flaging
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = true} : () -> i1
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 140733946907392 : ui64} : () -> ui64
    %5 = "daphne.createDaphneContext"(%4) : (ui64) -> !daphne.DaphneContext
    %6 = "daphne.fill"(%0, %3, %3) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %7 = "daphne.ewAdd"(%6, %0) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    %8 = "daphne.ewSqrt"(%7) {updateInPlace = true} : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.print"(%8, %1, %2) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after managing object references
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = true} : () -> i1
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 140733946907392 : ui64} : () -> ui64
    %5 = "daphne.createDaphneContext"(%4) : (ui64) -> !daphne.DaphneContext
    %6 = "daphne.fill"(%0, %3, %3) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
    %7 = "daphne.ewAdd"(%6, %0) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    "daphne.decRef"(%6) : (!daphne.Matrix<2x2xf64>) -> ()
    %8 = "daphne.ewSqrt"(%7) {updateInPlace = true} : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.decRef"(%7) : (!daphne.Matrix<2x2xf64>) -> ()
    "daphne.print"(%8, %1, %2) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
    "daphne.decRef"(%8) : (!daphne.Matrix<2x2xf64>) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after kernel lowering
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = true} : () -> i1
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = 2 : index} : () -> index
    %4 = "daphne.constant"() {value = 140733946907392 : ui64} : () -> ui64
    %5 = "daphne.call_kernel"(%4) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
    %6 = "daphne.call_kernel"(%0, %3, %3, %5) {callee = "_fill__DenseMatrix_double__double__size_t__size_t"} : (f64, index, index, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64>
    %7 = "daphne.call_kernel"(%6, %0, %5) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__double"} : (!daphne.Matrix<2x2xf64>, f64, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64>
    "daphne.call_kernel"(%6, %5) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64>, !daphne.DaphneContext) -> ()
    %8 = "daphne.call_kernel"(%7, %5) {callee = "_ewSqrt__DenseMatrix_double__DenseMatrix_double", updateInPlace = true} : (!daphne.Matrix<2x2xf64>, !daphne.DaphneContext) -> !daphne.Matrix<2x2xf64>
    "daphne.call_kernel"(%7, %5) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%8, %1, %2, %5) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x2xf64>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%8, %5) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x2xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%5) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
    "daphne.return"() : () -> ()
  }
}
//IR after llvm lowering
module {
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_ewSqrt__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>)
  llvm.func @_fill__DenseMatrix_double__double__size_t__size_t(!llvm.ptr<ptr<i1>>, f64, i64, i64, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(2 : index) : i64
    %4 = llvm.mlir.constant(140733946907392 : ui64) : i64
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %7 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %7, %6 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%6, %4) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %8 = llvm.load %6 : !llvm.ptr<ptr<i1>>
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.alloca %9 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %11 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %11, %10 : !llvm.ptr<ptr<i1>>
    llvm.call @_fill__DenseMatrix_double__double__size_t__size_t(%10, %0, %3, %3, %8) : (!llvm.ptr<ptr<i1>>, f64, i64, i64, !llvm.ptr<i1>) -> ()
    %12 = llvm.load %10 : !llvm.ptr<ptr<i1>>
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.alloca %13 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %15 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %15, %14 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__double(%14, %12, %0, %8) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>) -> ()
    %16 = llvm.load %14 : !llvm.ptr<ptr<i1>>
    %17 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%12, %8) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %18 = llvm.mlir.constant(1 : i64) : i64
    %19 = llvm.alloca %18 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %20 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %20, %19 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewSqrt__DenseMatrix_double__DenseMatrix_double(%19, %16, %8) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %21 = llvm.load %19 : !llvm.ptr<ptr<i1>>
    %22 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%16, %8) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %23 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%21, %1, %2, %8) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %24 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%21, %8) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %25 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%8) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}
