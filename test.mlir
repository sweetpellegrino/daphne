IR after update in place flagging:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = -2.000000e+00 : f64} : () -> f64
    %2 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64
    %4 = "daphne.constant"() {value = false} : () -> i1
    %5 = "daphne.constant"() {value = true} : () -> i1
    %6 = "daphne.constant"() {value = 2 : index} : () -> index
    %7 = "daphne.constant"() {value = 5 : index} : () -> index
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %10 = "daphne.constant"() {value = 11 : index} : () -> index
    %11 = "daphne.constant"() {value = 1 : index} : () -> index
    %12 = "daphne.constant"() {value = 140722950106608 : ui64} : () -> ui64
    %13 = "daphne.createDaphneContext"(%12) : (ui64) -> !daphne.DaphneContext
    %14 = "daphne.randMatrix"(%7, %7, %3, %8, %8, %2) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    %15 = "daphne.randMatrix"(%6, %7, %3, %8, %8, %2) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x5xf64:sp[1.000000e+00]>
    %16 = "daphne.cast"(%15) : (!daphne.Matrix<2x5xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x5xf64>
    %17 = scf.for %arg0 = %11 to %10 step %11 iter_args(%arg1 = %16) -> (!daphne.Matrix<2x5xf64>) {
      %18 = "daphne.cast"(%arg0) : (index) -> si64
      %19 = "daphne.ewMul"(%18, %9) : (si64, si64) -> si64
      %20 = "daphne.matMul"(%14, %arg1, %4, %5) : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.Matrix<2x5xf64>, i1, i1) -> !daphne.Matrix<5x2xf64>
      %21 = "daphne.ewMul"(%20, %1) : (!daphne.Matrix<5x2xf64>, f64) -> !daphne.Matrix<5x2xf64>
      %22 = "daphne.ewPow"(%arg1, %0) : (!daphne.Matrix<2x5xf64>, f64) -> !daphne.Matrix<2x5xf64>
      %23 = "daphne.sumRow"(%22) : (!daphne.Matrix<2x5xf64>) -> !daphne.Matrix<2x1xf64>
      %24 = "daphne.transpose"(%23) : (!daphne.Matrix<2x1xf64>) -> !daphne.Matrix<1x2xf64>
      %25 = "daphne.ewAdd"(%21, %24) : (!daphne.Matrix<5x2xf64>, !daphne.Matrix<1x2xf64>) -> !daphne.Matrix<5x2xf64>
      %26 = "daphne.minRow"(%25) : (!daphne.Matrix<5x2xf64>) -> !daphne.Matrix<5x1xf64>
      %27 = "daphne.ewLe"(%25, %26) : (!daphne.Matrix<5x2xf64>, !daphne.Matrix<5x1xf64>) -> !daphne.Matrix<5x2xf64>
      %28 = "daphne.sumRow"(%27) : (!daphne.Matrix<5x2xf64>) -> !daphne.Matrix<5x1xf64>
      %29 = "daphne.ewDiv"(%27, %28) : (!daphne.Matrix<5x2xf64>, !daphne.Matrix<5x1xf64>) -> !daphne.Matrix<5x2xf64>
      %30 = "daphne.sumCol"(%29) : (!daphne.Matrix<5x2xf64>) -> !daphne.Matrix<1x2xf64>
      %31 = "daphne.transpose"(%29) : (!daphne.Matrix<5x2xf64>) -> !daphne.Matrix<2x5xf64>
      %32 = "daphne.matMul"(%31, %14, %4, %4) : (!daphne.Matrix<2x5xf64>, !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<2x5xf64>
      %33 = "daphne.transpose"(%30) : (!daphne.Matrix<1x2xf64>) -> !daphne.Matrix<2x1xf64>
      %34 = "daphne.ewDiv"(%32, %33) : (!daphne.Matrix<2x5xf64>, !daphne.Matrix<2x1xf64>) -> !daphne.Matrix<2x5xf64>
      scf.yield %34 : !daphne.Matrix<2x5xf64>
    }
    "daphne.print"(%17, %5, %4) : (!daphne.Matrix<2x5xf64>, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}
IR after kernel lowering:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = -2.000000e+00 : f64} : () -> f64
    %2 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64
    %4 = "daphne.constant"() {value = false} : () -> i1
    %5 = "daphne.constant"() {value = true} : () -> i1
    %6 = "daphne.constant"() {value = 2 : index} : () -> index
    %7 = "daphne.constant"() {value = 5 : index} : () -> index
    %8 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %10 = "daphne.constant"() {value = 11 : index} : () -> index
    %11 = "daphne.constant"() {value = 1 : index} : () -> index
    %12 = "daphne.constant"() {value = 140722950106608 : ui64} : () -> ui64
    %13 = "daphne.call_kernel"(%12) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
    %14 = "daphne.call_kernel"(%7, %7, %3, %8, %8, %2, %13) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    %15 = "daphne.call_kernel"(%6, %7, %3, %8, %8, %2, %13) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<2x5xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%15, %13) {callee = "_incRef__Structure"} : (!daphne.Matrix<2x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    %16 = "daphne.cast"(%15) : (!daphne.Matrix<2x5xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x5xf64>
    "daphne.call_kernel"(%15, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %13) {callee = "_incRef__Structure"} : (!daphne.Matrix<2x5xf64>, !daphne.DaphneContext) -> ()
    %17 = scf.for %arg0 = %11 to %10 step %11 iter_args(%arg1 = %16) -> (!daphne.Matrix<2x5xf64>) {
      %18 = "daphne.call_kernel"(%arg0, %13) {callee = "_cast__int64_t__size_t"} : (index, !daphne.DaphneContext) -> si64
      %19 = "daphne.call_kernel"(%18, %9, %13) {callee = "_ewMul__int64_t__int64_t__int64_t"} : (si64, si64, !daphne.DaphneContext) -> si64
      %20 = "daphne.call_kernel"(%14, %arg1, %4, %5, %13) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.Matrix<2x5xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x2xf64>
      %21 = "daphne.call_kernel"(%20, %1, %13) {callee = "_ewMul__DenseMatrix_double__DenseMatrix_double__double"} : (!daphne.Matrix<5x2xf64>, f64, !daphne.DaphneContext) -> !daphne.Matrix<5x2xf64>
      "daphne.call_kernel"(%20, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> ()
      %22 = "daphne.call_kernel"(%arg1, %0, %13) {callee = "_ewPow__DenseMatrix_double__DenseMatrix_double__double"} : (!daphne.Matrix<2x5xf64>, f64, !daphne.DaphneContext) -> !daphne.Matrix<2x5xf64>
      "daphne.call_kernel"(%arg1, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x5xf64>, !daphne.DaphneContext) -> ()
      %23 = "daphne.call_kernel"(%22, %13) {callee = "_sumRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x5xf64>, !daphne.DaphneContext) -> !daphne.Matrix<2x1xf64>
      "daphne.call_kernel"(%22, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x5xf64>, !daphne.DaphneContext) -> ()
      %24 = "daphne.call_kernel"(%23, %13) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x1xf64>, !daphne.DaphneContext) -> !daphne.Matrix<1x2xf64>
      "daphne.call_kernel"(%23, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x1xf64>, !daphne.DaphneContext) -> ()
      %25 = "daphne.call_kernel"(%21, %24, %13) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x2xf64>, !daphne.Matrix<1x2xf64>, !daphne.DaphneContext) -> !daphne.Matrix<5x2xf64>
      "daphne.call_kernel"(%24, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<1x2xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%21, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> ()
      %26 = "daphne.call_kernel"(%25, %13) {callee = "_minRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> !daphne.Matrix<5x1xf64>
      %27 = "daphne.call_kernel"(%25, %26, %13) {callee = "_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x2xf64>, !daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> !daphne.Matrix<5x2xf64>
      "daphne.call_kernel"(%26, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%25, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> ()
      %28 = "daphne.call_kernel"(%27, %13) {callee = "_sumRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> !daphne.Matrix<5x1xf64>
      %29 = "daphne.call_kernel"(%27, %28, %13) {callee = "_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x2xf64>, !daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> !daphne.Matrix<5x2xf64>
      "daphne.call_kernel"(%28, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%27, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> ()
      %30 = "daphne.call_kernel"(%29, %13) {callee = "_sumCol__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> !daphne.Matrix<1x2xf64>
      %31 = "daphne.call_kernel"(%29, %13) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> !daphne.Matrix<2x5xf64>
      "daphne.call_kernel"(%29, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x2xf64>, !daphne.DaphneContext) -> ()
      %32 = "daphne.call_kernel"(%31, %14, %4, %4, %13) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x5xf64>, !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<2x5xf64>
      "daphne.call_kernel"(%31, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x5xf64>, !daphne.DaphneContext) -> ()
      %33 = "daphne.call_kernel"(%30, %13) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<1x2xf64>, !daphne.DaphneContext) -> !daphne.Matrix<2x1xf64>
      "daphne.call_kernel"(%30, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<1x2xf64>, !daphne.DaphneContext) -> ()
      %34 = "daphne.call_kernel"(%32, %33, %13) {callee = "_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<2x5xf64>, !daphne.Matrix<2x1xf64>, !daphne.DaphneContext) -> !daphne.Matrix<2x5xf64>
      "daphne.call_kernel"(%33, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x1xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%32, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x5xf64>, !daphne.DaphneContext) -> ()
      scf.yield %34 : !daphne.Matrix<2x5xf64>
    }
    "daphne.call_kernel"(%16, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x5xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%14, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%17, %5, %4, %13) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<2x5xf64>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%17, %13) {callee = "_decRef__Structure"} : (!daphne.Matrix<2x5xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%13) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
    "daphne.return"() : () -> ()
  }
}
IR after llvm lowering:
module {
  llvm.func @_sumCol__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_minRow__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_transpose__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_sumRow__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewPow__DenseMatrix_double__DenseMatrix_double__double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>)
  llvm.func @_ewMul__DenseMatrix_double__DenseMatrix_double__double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>)
  llvm.func @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_ewMul__int64_t__int64_t__int64_t(!llvm.ptr<i64>, i64, i64, !llvm.ptr<i1>)
  llvm.func @_cast__int64_t__size_t(!llvm.ptr<i64>, i64, !llvm.ptr<i1>)
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_incRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(-2.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(-1 : si64) : i64
    %3 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(true) : i1
    %6 = llvm.mlir.constant(2 : index) : i64
    %7 = llvm.mlir.constant(5 : index) : i64
    %8 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %9 = llvm.mlir.constant(1 : si64) : i64
    %10 = llvm.mlir.constant(11 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(140722950106608 : ui64) : i64
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.alloca %13 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %15 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %15, %14 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%14, %12) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %16 = llvm.load %14 : !llvm.ptr<ptr<i1>>
    %17 = llvm.mlir.constant(1 : i64) : i64
    %18 = llvm.alloca %17 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %19 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %19, %18 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%18, %7, %7, %3, %8, %8, %2, %16) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %20 = llvm.load %18 : !llvm.ptr<ptr<i1>>
    %21 = llvm.mlir.constant(1 : i64) : i64
    %22 = llvm.alloca %21 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %23 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %23, %22 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%22, %6, %7, %3, %8, %8, %2, %16) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %24 = llvm.load %22 : !llvm.ptr<ptr<i1>>
    %25 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_incRef__Structure(%24, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %26 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%24, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %27 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_incRef__Structure(%24, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    llvm.br ^bb1(%11, %24 : i64, !llvm.ptr<i1>)
  ^bb1(%28: i64, %29: !llvm.ptr<i1>):  // 2 preds: ^bb0, ^bb2
    %30 = llvm.icmp "slt" %28, %10 : i64
    llvm.cond_br %30, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %31 = llvm.mlir.constant(1 : i64) : i64
    %32 = llvm.alloca %31 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.call @_cast__int64_t__size_t(%32, %28, %16) : (!llvm.ptr<i64>, i64, !llvm.ptr<i1>) -> ()
    %33 = llvm.load %32 : !llvm.ptr<i64>
    %34 = llvm.mlir.constant(1 : i64) : i64
    %35 = llvm.alloca %34 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.call @_ewMul__int64_t__int64_t__int64_t(%35, %33, %9, %16) : (!llvm.ptr<i64>, i64, i64, !llvm.ptr<i1>) -> ()
    %36 = llvm.load %35 : !llvm.ptr<i64>
    %37 = llvm.mlir.constant(1 : i64) : i64
    %38 = llvm.alloca %37 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %39 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %39, %38 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%38, %20, %29, %4, %5, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %40 = llvm.load %38 : !llvm.ptr<ptr<i1>>
    %41 = llvm.mlir.constant(1 : i64) : i64
    %42 = llvm.alloca %41 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %43 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %43, %42 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewMul__DenseMatrix_double__DenseMatrix_double__double(%42, %40, %1, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>) -> ()
    %44 = llvm.load %42 : !llvm.ptr<ptr<i1>>
    %45 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%40, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %46 = llvm.mlir.constant(1 : i64) : i64
    %47 = llvm.alloca %46 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %48 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %48, %47 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewPow__DenseMatrix_double__DenseMatrix_double__double(%47, %29, %0, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>) -> ()
    %49 = llvm.load %47 : !llvm.ptr<ptr<i1>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%29, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %51 = llvm.mlir.constant(1 : i64) : i64
    %52 = llvm.alloca %51 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %53 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %53, %52 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumRow__DenseMatrix_double__DenseMatrix_double(%52, %49, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %54 = llvm.load %52 : !llvm.ptr<ptr<i1>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%49, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %56 = llvm.mlir.constant(1 : i64) : i64
    %57 = llvm.alloca %56 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %58 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %58, %57 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%57, %54, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %59 = llvm.load %57 : !llvm.ptr<ptr<i1>>
    %60 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%54, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %61 = llvm.mlir.constant(1 : i64) : i64
    %62 = llvm.alloca %61 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %63 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %63, %62 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%62, %44, %59, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %64 = llvm.load %62 : !llvm.ptr<ptr<i1>>
    %65 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%59, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %66 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%44, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %67 = llvm.mlir.constant(1 : i64) : i64
    %68 = llvm.alloca %67 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %69 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %69, %68 : !llvm.ptr<ptr<i1>>
    llvm.call @_minRow__DenseMatrix_double__DenseMatrix_double(%68, %64, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %70 = llvm.load %68 : !llvm.ptr<ptr<i1>>
    %71 = llvm.mlir.constant(1 : i64) : i64
    %72 = llvm.alloca %71 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %73 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %73, %72 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%72, %64, %70, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %74 = llvm.load %72 : !llvm.ptr<ptr<i1>>
    %75 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%70, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %76 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%64, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %77 = llvm.mlir.constant(1 : i64) : i64
    %78 = llvm.alloca %77 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %79 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %79, %78 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumRow__DenseMatrix_double__DenseMatrix_double(%78, %74, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %80 = llvm.load %78 : !llvm.ptr<ptr<i1>>
    %81 = llvm.mlir.constant(1 : i64) : i64
    %82 = llvm.alloca %81 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %83 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %83, %82 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%82, %74, %80, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %84 = llvm.load %82 : !llvm.ptr<ptr<i1>>
    %85 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%80, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %86 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%74, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %87 = llvm.mlir.constant(1 : i64) : i64
    %88 = llvm.alloca %87 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %89 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %89, %88 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumCol__DenseMatrix_double__DenseMatrix_double(%88, %84, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %90 = llvm.load %88 : !llvm.ptr<ptr<i1>>
    %91 = llvm.mlir.constant(1 : i64) : i64
    %92 = llvm.alloca %91 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %93 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %93, %92 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%92, %84, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %94 = llvm.load %92 : !llvm.ptr<ptr<i1>>
    %95 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%84, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %96 = llvm.mlir.constant(1 : i64) : i64
    %97 = llvm.alloca %96 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %98 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %98, %97 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%97, %94, %20, %4, %4, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %99 = llvm.load %97 : !llvm.ptr<ptr<i1>>
    %100 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%94, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %101 = llvm.mlir.constant(1 : i64) : i64
    %102 = llvm.alloca %101 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %103 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %103, %102 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%102, %90, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %104 = llvm.load %102 : !llvm.ptr<ptr<i1>>
    %105 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%90, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %106 = llvm.mlir.constant(1 : i64) : i64
    %107 = llvm.alloca %106 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %108 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %108, %107 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%107, %99, %104, %16) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %109 = llvm.load %107 : !llvm.ptr<ptr<i1>>
    %110 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%104, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %111 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%99, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %112 = llvm.add %28, %11  : i64
    llvm.br ^bb1(%112, %109 : i64, !llvm.ptr<i1>)
  ^bb3:  // pred: ^bb1
    %113 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%24, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %114 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%20, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %115 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%29, %5, %4, %16) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %116 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%29, %16) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %117 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%16) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}