module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = -2.000000e+00 : f64} : () -> f64
    %2 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64
    %4 = "daphne.constant"() {value = false} : () -> i1
    %5 = "daphne.constant"() {value = true} : () -> i1
    %6 = "daphne.constant"() {value = 5 : index} : () -> index
    %7 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %8 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %9 = "daphne.constant"() {value = 6 : index} : () -> index
    %10 = "daphne.constant"() {value = 1 : index} : () -> index
    %11 = "daphne.constant"() {value = 140736352119160 : ui64} : () -> ui64
    %12 = "daphne.call_kernel"(%11) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
    %13 = "daphne.call_kernel"(%6, %6, %3, %7, %7, %2, %12) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    %14 = "daphne.call_kernel"(%6, %6, %3, %7, %7, %2, %12) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%14, %12) {callee = "_incRef__Structure"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    %15 = "daphne.cast"(%14) : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>) -> !daphne.Matrix<5x5xf64>
    "daphne.call_kernel"(%14, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%15, %12) {callee = "_incRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
    %16 = scf.for %arg0 = %10 to %9 step %10 iter_args(%arg1 = %15) -> (!daphne.Matrix<5x5xf64>) {
      %17 = "daphne.call_kernel"(%arg0, %12) {callee = "_cast__int64_t__size_t"} : (index, !daphne.DaphneContext) -> si64
      %18 = "daphne.call_kernel"(%17, %8, %12) {callee = "_ewMul__int64_t__int64_t__int64_t"} : (si64, si64, !daphne.DaphneContext) -> si64
      %19 = "daphne.call_kernel"(%13, %arg1, %4, %5, %12) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.Matrix<5x5xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      %20 = "daphne.constant"() {value = false} : () -> i1
      %21 = "daphne.call_kernel"(%19, %1, %20, %12) {callee = "_ewMul__DenseMatrix_double__DenseMatrix_double__double__bool"} : (!daphne.Matrix<5x5xf64>, f64, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%19, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %22 = "daphne.constant"() {value = false} : () -> i1
      %23 = "daphne.call_kernel"(%arg1, %0, %22, %12) {callee = "_ewPow__DenseMatrix_double__DenseMatrix_double__double__bool"} : (!daphne.Matrix<5x5xf64>, f64, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%arg1, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %24 = "daphne.call_kernel"(%23, %12) {callee = "_sumRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> !daphne.Matrix<5x1xf64>
      "daphne.call_kernel"(%23, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %25 = "daphne.constant"() {value = false} : () -> i1
      %26 = "daphne.call_kernel"(%24, %25, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double__bool"} : (!daphne.Matrix<5x1xf64>, i1, !daphne.DaphneContext) -> !daphne.Matrix<1x5xf64>
      "daphne.call_kernel"(%24, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> ()
      %27 = "daphne.constant"() {value = false} : () -> i1
      %28 = "daphne.constant"() {value = false} : () -> i1
      %29 = "daphne.call_kernel"(%21, %26, %27, %28, %12) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<1x5xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%26, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<1x5xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%21, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %30 = "daphne.call_kernel"(%29, %12) {callee = "_minRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> !daphne.Matrix<5x1xf64>
      %31 = "daphne.constant"() {value = false} : () -> i1
      %32 = "daphne.constant"() {value = false} : () -> i1
      %33 = "daphne.call_kernel"(%29, %30, %31, %32, %12) {callee = "_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x1xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%30, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%29, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %34 = "daphne.call_kernel"(%33, %12) {callee = "_sumRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> !daphne.Matrix<5x1xf64>
      %35 = "daphne.constant"() {value = false} : () -> i1
      %36 = "daphne.constant"() {value = false} : () -> i1
      %37 = "daphne.call_kernel"(%33, %34, %35, %36, %12) {callee = "_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x1xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%34, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%33, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %38 = "daphne.call_kernel"(%37, %12) {callee = "_sumCol__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> !daphne.Matrix<1x5xf64>
      %39 = "daphne.constant"() {value = false} : () -> i1
      %40 = "daphne.call_kernel"(%37, %39, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double__bool"} : (!daphne.Matrix<5x5xf64>, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%37, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %41 = "daphne.call_kernel"(%40, %13, %4, %4, %12) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%40, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %42 = "daphne.constant"() {value = false} : () -> i1
      %43 = "daphne.call_kernel"(%38, %42, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double__bool"} : (!daphne.Matrix<1x5xf64>, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x1xf64>
      "daphne.call_kernel"(%38, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<1x5xf64>, !daphne.DaphneContext) -> ()
      %44 = "daphne.constant"() {value = false} : () -> i1
      %45 = "daphne.constant"() {value = false} : () -> i1
      %46 = "daphne.call_kernel"(%41, %43, %44, %45, %12) {callee = "_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x1xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%43, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%41, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      scf.yield %46 : !daphne.Matrix<5x5xf64>
    }
    "daphne.call_kernel"(%15, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%13, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %5, %4, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%12) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
    "daphne.return"() : () -> ()
  }
}