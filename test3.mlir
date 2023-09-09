%23 = "daphne.ewPow"(%arg2, %arg3) : (!daphne.Matrix<?x5xf64>, f64) -> !daphne.Matrix<?x?xf64>%25 = "daphne.transpose"(%24) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%24 = "daphne.ewMul"(%23, %arg6) : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>%25 = "daphne.ewAdd"(%24, %arg7) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x5xf64>) -> !daphne.Matrix<?x?xf64>%25 = "daphne.ewAdd"(%24, %arg7) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x5xf64>) -> !daphne.Matrix<?x?xf64>%27 = "daphne.ewLe"(%25, %26) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%27 = "daphne.ewLe"(%25, %26) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%27 = "daphne.ewLe"(%25, %26) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%29 = "daphne.ewDiv"(%27, %28) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%29 = "daphne.ewDiv"(%27, %28) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%29 = "daphne.ewDiv"(%27, %28) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%30 = "daphne.transpose"(%29) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%30 = "daphne.transpose"(%29) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>%23 = "daphne.transpose"(%arg2) : (!daphne.Matrix<?x5xf64>) -> !daphne.Matrix<?x?xf64>%24 = "daphne.ewDiv"(%23, %arg5) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64>%24 = "daphne.ewDiv"(%23, %arg5) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64>IR after update in place flagging:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 5 : index} : () -> index
    %5 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %6 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 6 : index} : () -> index
    %8 = "daphne.constant"() {value = 1 : index} : () -> index
    %9 = "daphne.constant"() {value = -2.000000e+00 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 2.000000e+00 : f64} : () -> f64
    %11 = "daphne.constant"() {value = 140723330350072 : ui64} : () -> ui64
    %12 = "daphne.createDaphneContext"(%11) : (ui64) -> !daphne.DaphneContext
    %13 = "daphne.randMatrix"(%4, %4, %1, %5, %5, %0) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    %14 = "daphne.randMatrix"(%4, %4, %1, %5, %5, %0) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    %15 = "daphne.cast"(%14) : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>) -> !daphne.Matrix<5x5xf64>
    %16 = scf.for %arg0 = %8 to %7 step %8 iter_args(%arg1 = %15) -> (!daphne.Matrix<5x5xf64>) {
      %17 = "daphne.cast"(%arg0) : (index) -> si64
      %18 = "daphne.ewMul"(%17, %6) {inPlaceFutureUse = [true, true]} : (si64, si64) -> si64
      %19 = "daphne.vectorizedPipeline"(%arg1, %10, %8, %4) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64>, %arg3: f64):
        %23 = "daphne.ewPow"(%arg2, %arg3) {inPlaceFutureUse = [false, true]} : (!daphne.Matrix<?x5xf64>, f64) -> !daphne.Matrix<?x?xf64>
        %24 = "daphne.sumRow"(%23) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %25 = "daphne.transpose"(%24) {inPlaceFutureUse = [false]} : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        "daphne.return"(%25) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2], operand_segment_sizes = array<i32: 2, 1, 1, 0>, splits = [1, 0]} : (!daphne.Matrix<5x5xf64>, f64, index, index) -> !daphne.Matrix<1x5xf64>
      %20:2 = "daphne.vectorizedPipeline"(%13, %arg1, %2, %3, %9, %19, %4, %8, %4, %4) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64:sp[1.000000e+00]>, %arg3: !daphne.Matrix<5x5xf64>, %arg4: i1, %arg5: i1, %arg6: f64, %arg7: !daphne.Matrix<?x5xf64>):
        %23 = "daphne.matMul"(%arg2, %arg3, %arg4, %arg5) : (!daphne.Matrix<?x5xf64:sp[1.000000e+00]>, !daphne.Matrix<5x5xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
        %24 = "daphne.ewMul"(%23, %arg6) {inPlaceFutureUse = [false, true]} : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
        %25 = "daphne.ewAdd"(%24, %arg7) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x5xf64>) -> !daphne.Matrix<?x?xf64>
        %26 = "daphne.minRow"(%25) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %27 = "daphne.ewLe"(%25, %26) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %28 = "daphne.sumRow"(%27) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %29 = "daphne.ewDiv"(%27, %28) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %30 = "daphne.transpose"(%29) {inPlaceFutureUse = [true]} : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %31 = "daphne.sumCol"(%29) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        "daphne.return"(%30, %31) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2, 3], operand_segment_sizes = array<i32: 6, 2, 2, 0>, splits = [1, 0, 0, 0, 0, 1]} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.Matrix<5x5xf64>, i1, i1, f64, !daphne.Matrix<1x5xf64>, index, index, index, index) -> (!daphne.Matrix<5x5xf64>, !daphne.Matrix<1x5xf64>)
      %21 = "daphne.vectorizedPipeline"(%20#1, %4, %8) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64>):
        %23 = "daphne.transpose"(%arg2) {inPlaceFutureUse = [false]} : (!daphne.Matrix<?x5xf64>) -> !daphne.Matrix<?x?xf64>
        "daphne.return"(%23) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2], operand_segment_sizes = array<i32: 1, 1, 1, 0>, splits = [1]} : (!daphne.Matrix<1x5xf64>, index, index) -> !daphne.Matrix<5x1xf64>
      %22 = "daphne.vectorizedPipeline"(%20#0, %13, %2, %21, %4, %4) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64>, %arg3: !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, %arg4: i1, %arg5: !daphne.Matrix<?x1xf64>):
        %23 = "daphne.matMul"(%arg2, %arg3, %arg4, %arg4) : (!daphne.Matrix<?x5xf64>, !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<?x?xf64>
        %24 = "daphne.ewDiv"(%23, %arg5) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64>
        "daphne.return"(%24) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [1], operand_segment_sizes = array<i32: 4, 1, 1, 0>, splits = [1, 0, 0, 1]} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, i1, !daphne.Matrix<5x1xf64>, index, index) -> !daphne.Matrix<5x5xf64>
      scf.yield %22 : !daphne.Matrix<5x5xf64>
    }
    "daphne.print"(%16, %3, %2) : (!daphne.Matrix<5x5xf64>, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}
IR after kernel lowering:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 5 : index} : () -> index
    %5 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %6 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 6 : index} : () -> index
    %8 = "daphne.constant"() {value = 1 : index} : () -> index
    %9 = "daphne.constant"() {value = -2.000000e+00 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 2.000000e+00 : f64} : () -> f64
    %11 = "daphne.constant"() {value = 140723330350072 : ui64} : () -> ui64
    %12 = "daphne.call_kernel"(%11) {callee = "_createDaphneContext__DaphneContext__uint64_t"} : (ui64) -> !daphne.DaphneContext
    %13 = "daphne.call_kernel"(%4, %4, %1, %5, %5, %0, %12) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    %14 = "daphne.call_kernel"(%4, %4, %1, %5, %5, %0, %12) {callee = "_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t"} : (index, index, f64, f64, f64, si64, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    "daphne.call_kernel"(%14, %12) {callee = "_incRef__Structure"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    %15 = "daphne.cast"(%14) : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>) -> !daphne.Matrix<5x5xf64>
    "daphne.call_kernel"(%14, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%15, %12) {callee = "_incRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
    %16 = scf.for %arg0 = %8 to %7 step %8 iter_args(%arg1 = %15) -> (!daphne.Matrix<5x5xf64>) {
      %17 = "daphne.call_kernel"(%arg0, %12) {callee = "_cast__int64_t__size_t"} : (index, !daphne.DaphneContext) -> si64
      %18 = "daphne.call_kernel"(%17, %6, %12) {callee = "_ewMul__int64_t__int64_t__int64_t"} : (si64, si64, !daphne.DaphneContext) -> si64
      %19 = "daphne.vectorizedPipeline"(%arg1, %10, %8, %4, %12) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64>, %arg3: f64):
        %23 = "daphne.constant"() {value = false} : () -> i1
        %24 = "daphne.call_kernel"(%arg2, %arg3, %23, %12) {callee = "_ewPow__DenseMatrix_double__DenseMatrix_double__double__bool"} : (!daphne.Matrix<?x5xf64>, f64, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%arg2, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x5xf64>, !daphne.DaphneContext) -> ()
        %25 = "daphne.call_kernel"(%24, %12) {callee = "_sumRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%24, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        %26 = "daphne.constant"() {value = false} : () -> i1
        %27 = "daphne.call_kernel"(%25, %26, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double__bool"} : (!daphne.Matrix<?x?xf64>, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%25, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        "daphne.return"(%27) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2], operand_segment_sizes = array<i32: 2, 1, 1, 1>, splits = [1, 0]} : (!daphne.Matrix<5x5xf64>, f64, index, index, !daphne.DaphneContext) -> !daphne.Matrix<1x5xf64>
      %20:2 = "daphne.vectorizedPipeline"(%13, %arg1, %2, %3, %9, %19, %4, %8, %4, %4, %12) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64:sp[1.000000e+00]>, %arg3: !daphne.Matrix<5x5xf64>, %arg4: i1, %arg5: i1, %arg6: f64, %arg7: !daphne.Matrix<?x5xf64>):
        %23 = "daphne.call_kernel"(%arg2, %arg3, %arg4, %arg5, %12) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<?x5xf64:sp[1.000000e+00]>, !daphne.Matrix<5x5xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%arg3, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
        "daphne.call_kernel"(%arg2, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
        %24 = "daphne.constant"() {value = false} : () -> i1
        %25 = "daphne.call_kernel"(%23, %arg6, %24, %12) {callee = "_ewMul__DenseMatrix_double__DenseMatrix_double__double__bool"} : (!daphne.Matrix<?x?xf64>, f64, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%23, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        %26 = "daphne.constant"() {value = false} : () -> i1
        %27 = "daphne.constant"() {value = false} : () -> i1
        %28 = "daphne.call_kernel"(%25, %arg7, %26, %27, %12) {callee = "_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x5xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%25, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        "daphne.call_kernel"(%arg7, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x5xf64>, !daphne.DaphneContext) -> ()
        %29 = "daphne.call_kernel"(%28, %12) {callee = "_minRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        %30 = "daphne.constant"() {value = false} : () -> i1
        %31 = "daphne.constant"() {value = false} : () -> i1
        %32 = "daphne.call_kernel"(%28, %29, %30, %31, %12) {callee = "_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%29, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        "daphne.call_kernel"(%28, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        %33 = "daphne.call_kernel"(%32, %12) {callee = "_sumRow__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        %34 = "daphne.constant"() {value = false} : () -> i1
        %35 = "daphne.constant"() {value = false} : () -> i1
        %36 = "daphne.call_kernel"(%32, %33, %34, %35, %12) {callee = "_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%33, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        "daphne.call_kernel"(%32, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        %37 = "daphne.constant"() {value = true} : () -> i1
        %38 = "daphne.call_kernel"(%36, %37, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double__bool"} : (!daphne.Matrix<?x?xf64>, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        %39 = "daphne.call_kernel"(%36, %12) {callee = "_sumCol__DenseMatrix_double__DenseMatrix_double"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%36, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        "daphne.return"(%38, %39) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2, 3], operand_segment_sizes = array<i32: 6, 2, 2, 1>, splits = [1, 0, 0, 0, 0, 1]} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.Matrix<5x5xf64>, i1, i1, f64, !daphne.Matrix<1x5xf64>, index, index, index, index, !daphne.DaphneContext) -> (!daphne.Matrix<5x5xf64>, !daphne.Matrix<1x5xf64>)
      "daphne.call_kernel"(%19, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<1x5xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%arg1, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      %21 = "daphne.vectorizedPipeline"(%20#1, %4, %8, %12) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64>):
        %23 = "daphne.constant"() {value = false} : () -> i1
        %24 = "daphne.call_kernel"(%arg2, %23, %12) {callee = "_transpose__DenseMatrix_double__DenseMatrix_double__bool"} : (!daphne.Matrix<?x5xf64>, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%arg2, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x5xf64>, !daphne.DaphneContext) -> ()
        "daphne.return"(%24) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2], operand_segment_sizes = array<i32: 1, 1, 1, 1>, splits = [1]} : (!daphne.Matrix<1x5xf64>, index, index, !daphne.DaphneContext) -> !daphne.Matrix<5x1xf64>
      "daphne.call_kernel"(%20#1, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<1x5xf64>, !daphne.DaphneContext) -> ()
      %22 = "daphne.vectorizedPipeline"(%20#0, %13, %2, %21, %4, %4, %12) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64>, %arg3: !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, %arg4: i1, %arg5: !daphne.Matrix<?x1xf64>):
        %23 = "daphne.call_kernel"(%arg2, %arg3, %arg4, %arg4, %12) {callee = "_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<?x5xf64>, !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%arg3, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
        "daphne.call_kernel"(%arg2, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x5xf64>, !daphne.DaphneContext) -> ()
        %24 = "daphne.constant"() {value = false} : () -> i1
        %25 = "daphne.constant"() {value = false} : () -> i1
        %26 = "daphne.call_kernel"(%23, %arg5, %24, %25, %12) {callee = "_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x1xf64>, i1, i1, !daphne.DaphneContext) -> !daphne.Matrix<?x?xf64>
        "daphne.call_kernel"(%23, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x?xf64>, !daphne.DaphneContext) -> ()
        "daphne.call_kernel"(%arg5, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<?x1xf64>, !daphne.DaphneContext) -> ()
        "daphne.return"(%26) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [1], operand_segment_sizes = array<i32: 4, 1, 1, 1>, splits = [1, 0, 0, 1]} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, i1, !daphne.Matrix<5x1xf64>, index, index, !daphne.DaphneContext) -> !daphne.Matrix<5x5xf64>
      "daphne.call_kernel"(%21, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x1xf64>, !daphne.DaphneContext) -> ()
      "daphne.call_kernel"(%20#0, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
      scf.yield %22 : !daphne.Matrix<5x5xf64>
    }
    "daphne.call_kernel"(%15, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%13, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %3, %2, %12) {callee = "_print__DenseMatrix_double__bool__bool"} : (!daphne.Matrix<5x5xf64>, i1, i1, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%16, %12) {callee = "_decRef__Structure"} : (!daphne.Matrix<5x5xf64>, !daphne.DaphneContext) -> ()
    "daphne.call_kernel"(%12) {callee = "_destroyDaphneContext"} : (!daphne.DaphneContext) -> ()
    "daphne.return"() : () -> ()
  }
}
IR after llvm lowering:
module {
  llvm.func @_vect4(%arg0: !llvm.ptr<ptr<ptr<i1>>>, %arg1: !llvm.ptr<ptr<i1>>, %arg2: !llvm.ptr<i1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.load %arg1 : !llvm.ptr<ptr<i1>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.getelementptr %arg1[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %4 = llvm.load %3 : !llvm.ptr<ptr<i1>>
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.getelementptr %arg1[2] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %7 = llvm.load %6 : !llvm.ptr<ptr<i1>>
    %8 = llvm.ptrtoint %7 : !llvm.ptr<i1> to i64
    %9 = llvm.trunc %8 : i64 to i1
    %10 = llvm.mlir.constant(3 : i64) : i64
    %11 = llvm.getelementptr %arg1[3] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %12 = llvm.load %11 : !llvm.ptr<ptr<i1>>
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.alloca %13 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %15 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %15, %14 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%14, %1, %4, %9, %9, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %16 = llvm.load %14 : !llvm.ptr<ptr<i1>>
    %17 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%4, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %18 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%1, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %19 = llvm.mlir.constant(false) : i1
    %20 = llvm.mlir.constant(false) : i1
    %21 = llvm.mlir.constant(1 : i64) : i64
    %22 = llvm.alloca %21 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %23 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %23, %22 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%22, %16, %12, %19, %20, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %24 = llvm.load %22 : !llvm.ptr<ptr<i1>>
    %25 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%16, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %26 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%12, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %27 = llvm.mlir.constant(0 : i64) : i64
    %28 = llvm.load %arg0 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %24, %28 : !llvm.ptr<ptr<i1>>
    llvm.return
  }
  llvm.func @_vect3(%arg0: !llvm.ptr<ptr<ptr<i1>>>, %arg1: !llvm.ptr<ptr<i1>>, %arg2: !llvm.ptr<i1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.load %arg1 : !llvm.ptr<ptr<i1>>
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %5 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %5, %4 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double__bool(%4, %1, %2, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, i1, !llvm.ptr<i1>) -> ()
    %6 = llvm.load %4 : !llvm.ptr<ptr<i1>>
    %7 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%1, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.load %arg0 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %6, %9 : !llvm.ptr<ptr<i1>>
    llvm.return
  }
  llvm.func @_sumCol__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_minRow__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_ewMul__DenseMatrix_double__DenseMatrix_double__double__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, i1, !llvm.ptr<i1>)
  llvm.func @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_vect2(%arg0: !llvm.ptr<ptr<ptr<i1>>>, %arg1: !llvm.ptr<ptr<i1>>, %arg2: !llvm.ptr<i1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.load %arg1 : !llvm.ptr<ptr<i1>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.getelementptr %arg1[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %4 = llvm.load %3 : !llvm.ptr<ptr<i1>>
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.getelementptr %arg1[2] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %7 = llvm.load %6 : !llvm.ptr<ptr<i1>>
    %8 = llvm.ptrtoint %7 : !llvm.ptr<i1> to i64
    %9 = llvm.trunc %8 : i64 to i1
    %10 = llvm.mlir.constant(3 : i64) : i64
    %11 = llvm.getelementptr %arg1[3] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %12 = llvm.load %11 : !llvm.ptr<ptr<i1>>
    %13 = llvm.ptrtoint %12 : !llvm.ptr<i1> to i64
    %14 = llvm.trunc %13 : i64 to i1
    %15 = llvm.mlir.constant(4 : i64) : i64
    %16 = llvm.getelementptr %arg1[4] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %17 = llvm.load %16 : !llvm.ptr<ptr<i1>>
    %18 = llvm.ptrtoint %17 : !llvm.ptr<i1> to i64
    %19 = llvm.bitcast %18 : i64 to f64
    %20 = llvm.fptrunc %19 : f64 to f64
    %21 = llvm.mlir.constant(5 : i64) : i64
    %22 = llvm.getelementptr %arg1[5] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %23 = llvm.load %22 : !llvm.ptr<ptr<i1>>
    %24 = llvm.mlir.constant(1 : i64) : i64
    %25 = llvm.alloca %24 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %26 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %26, %25 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%25, %1, %4, %9, %14, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %27 = llvm.load %25 : !llvm.ptr<ptr<i1>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%4, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %29 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%1, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %30 = llvm.mlir.constant(false) : i1
    %31 = llvm.mlir.constant(1 : i64) : i64
    %32 = llvm.alloca %31 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %33 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %33, %32 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewMul__DenseMatrix_double__DenseMatrix_double__double__bool(%32, %27, %20, %30, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, i1, !llvm.ptr<i1>) -> ()
    %34 = llvm.load %32 : !llvm.ptr<ptr<i1>>
    %35 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%27, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %36 = llvm.mlir.constant(false) : i1
    %37 = llvm.mlir.constant(false) : i1
    %38 = llvm.mlir.constant(1 : i64) : i64
    %39 = llvm.alloca %38 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %40 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %40, %39 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%39, %34, %23, %36, %37, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %41 = llvm.load %39 : !llvm.ptr<ptr<i1>>
    %42 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%34, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %43 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%23, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %44 = llvm.mlir.constant(1 : i64) : i64
    %45 = llvm.alloca %44 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %46 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %46, %45 : !llvm.ptr<ptr<i1>>
    llvm.call @_minRow__DenseMatrix_double__DenseMatrix_double(%45, %41, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %47 = llvm.load %45 : !llvm.ptr<ptr<i1>>
    %48 = llvm.mlir.constant(false) : i1
    %49 = llvm.mlir.constant(false) : i1
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.alloca %50 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %52 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %52, %51 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%51, %41, %47, %48, %49, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %53 = llvm.load %51 : !llvm.ptr<ptr<i1>>
    %54 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%47, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %55 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%41, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %56 = llvm.mlir.constant(1 : i64) : i64
    %57 = llvm.alloca %56 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %58 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %58, %57 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumRow__DenseMatrix_double__DenseMatrix_double(%57, %53, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %59 = llvm.load %57 : !llvm.ptr<ptr<i1>>
    %60 = llvm.mlir.constant(false) : i1
    %61 = llvm.mlir.constant(false) : i1
    %62 = llvm.mlir.constant(1 : i64) : i64
    %63 = llvm.alloca %62 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %64 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %64, %63 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%63, %53, %59, %60, %61, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %65 = llvm.load %63 : !llvm.ptr<ptr<i1>>
    %66 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%59, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %67 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%53, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %68 = llvm.mlir.constant(true) : i1
    %69 = llvm.mlir.constant(1 : i64) : i64
    %70 = llvm.alloca %69 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %71 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %71, %70 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double__bool(%70, %65, %68, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, i1, !llvm.ptr<i1>) -> ()
    %72 = llvm.load %70 : !llvm.ptr<ptr<i1>>
    %73 = llvm.mlir.constant(1 : i64) : i64
    %74 = llvm.alloca %73 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %75 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %75, %74 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumCol__DenseMatrix_double__DenseMatrix_double(%74, %65, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %76 = llvm.load %74 : !llvm.ptr<ptr<i1>>
    %77 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%65, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %78 = llvm.mlir.constant(0 : i64) : i64
    %79 = llvm.load %arg0 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %72, %79 : !llvm.ptr<ptr<i1>>
    %80 = llvm.mlir.constant(1 : i64) : i64
    %81 = llvm.getelementptr %arg0[1] : (!llvm.ptr<ptr<ptr<i1>>>) -> !llvm.ptr<ptr<ptr<i1>>>
    %82 = llvm.load %81 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %76, %82 : !llvm.ptr<ptr<i1>>
    llvm.return
  }
  llvm.func @_transpose__DenseMatrix_double__DenseMatrix_double__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, i1, !llvm.ptr<i1>)
  llvm.func @_sumRow__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewPow__DenseMatrix_double__DenseMatrix_double__double__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, i1, !llvm.ptr<i1>)
  llvm.func @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>)
  llvm.func @_vect1(%arg0: !llvm.ptr<ptr<ptr<i1>>>, %arg1: !llvm.ptr<ptr<i1>>, %arg2: !llvm.ptr<i1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.load %arg1 : !llvm.ptr<ptr<i1>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.getelementptr %arg1[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %4 = llvm.load %3 : !llvm.ptr<ptr<i1>>
    %5 = llvm.ptrtoint %4 : !llvm.ptr<i1> to i64
    %6 = llvm.bitcast %5 : i64 to f64
    %7 = llvm.fptrunc %6 : f64 to f64
    %8 = llvm.mlir.constant(false) : i1
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.alloca %9 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %11 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %11, %10 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewPow__DenseMatrix_double__DenseMatrix_double__double__bool(%10, %1, %7, %8, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, i1, !llvm.ptr<i1>) -> ()
    %12 = llvm.load %10 : !llvm.ptr<ptr<i1>>
    %13 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%1, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.alloca %14 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %16 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %16, %15 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumRow__DenseMatrix_double__DenseMatrix_double(%15, %12, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %17 = llvm.load %15 : !llvm.ptr<ptr<i1>>
    %18 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%12, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %19 = llvm.mlir.constant(false) : i1
    %20 = llvm.mlir.constant(1 : i64) : i64
    %21 = llvm.alloca %20 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %22 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %22, %21 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double__bool(%21, %17, %19, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, i1, !llvm.ptr<i1>) -> ()
    %23 = llvm.load %21 : !llvm.ptr<ptr<i1>>
    %24 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%17, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %25 = llvm.mlir.constant(0 : i64) : i64
    %26 = llvm.load %arg0 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %23, %26 : !llvm.ptr<ptr<i1>>
    llvm.return
  }
  llvm.func @_ewMul__int64_t__int64_t__int64_t(!llvm.ptr<i64>, i64, i64, !llvm.ptr<i1>)
  llvm.func @_cast__int64_t__size_t(!llvm.ptr<i64>, i64, !llvm.ptr<i1>)
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_incRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(-1 : si64) : i64
    %1 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.constant(5 : index) : i64
    %5 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(1 : si64) : i64
    %7 = llvm.mlir.constant(6 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(-2.000000e+00 : f64) : f64
    %10 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %11 = llvm.mlir.constant(140723330350072 : ui64) : i64
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.alloca %12 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %14 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %14, %13 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%13, %11) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %15 = llvm.load %13 : !llvm.ptr<ptr<i1>>
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %18 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %18, %17 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%17, %4, %4, %1, %5, %5, %0, %15) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %19 = llvm.load %17 : !llvm.ptr<ptr<i1>>
    %20 = llvm.mlir.constant(1 : i64) : i64
    %21 = llvm.alloca %20 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %22 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %22, %21 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%21, %4, %4, %1, %5, %5, %0, %15) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %23 = llvm.load %21 : !llvm.ptr<ptr<i1>>
    %24 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_incRef__Structure(%23, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %25 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%23, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %26 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_incRef__Structure(%23, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    llvm.br ^bb1(%8, %23 : i64, !llvm.ptr<i1>)
  ^bb1(%27: i64, %28: !llvm.ptr<i1>):  // 2 preds: ^bb0, ^bb2
    %29 = llvm.icmp "slt" %27, %7 : i64
    llvm.cond_br %29, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %30 = llvm.mlir.constant(1 : i64) : i64
    %31 = llvm.alloca %30 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.call @_cast__int64_t__size_t(%31, %27, %15) : (!llvm.ptr<i64>, i64, !llvm.ptr<i1>) -> ()
    %32 = llvm.load %31 : !llvm.ptr<i64>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.alloca %33 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.call @_ewMul__int64_t__int64_t__int64_t(%34, %32, %6, %15) : (!llvm.ptr<i64>, i64, i64, !llvm.ptr<i1>) -> ()
    %35 = llvm.load %34 : !llvm.ptr<i64>
    %36 = llvm.mlir.addressof @_vect1 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>>
    %37 = llvm.mlir.constant(2 : i64) : i64
    %38 = llvm.alloca %37 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i1>
    %39 = llvm.mlir.constant(2 : i64) : i64
    %40 = llvm.alloca %39 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %41 = llvm.mlir.constant(false) : i1
    %42 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %41, %38 : !llvm.ptr<i1>
    %43 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %28, %40 : !llvm.ptr<ptr<i1>>
    %44 = llvm.mlir.constant(true) : i1
    %45 = llvm.mlir.constant(1 : i64) : i64
    %46 = llvm.getelementptr %38[1] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %44, %46 : !llvm.ptr<i1>
    %47 = llvm.mlir.constant(1 : i64) : i64
    %48 = llvm.getelementptr %40[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %49 = llvm.fpext %10 : f64 to f64
    %50 = llvm.bitcast %49 : f64 to i64
    %51 = llvm.inttoptr %50 : i64 to !llvm.ptr<i1>
    llvm.store %51, %48 : !llvm.ptr<ptr<i1>>
    %52 = llvm.mlir.constant(2 : index) : i64
    %53 = llvm.mlir.constant(1 : i64) : i64
    %54 = llvm.alloca %53 x i64 : (i64) -> !llvm.ptr<i64>
    %55 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %8, %54 : !llvm.ptr<i64>
    %56 = llvm.mlir.constant(1 : i64) : i64
    %57 = llvm.alloca %56 x i64 : (i64) -> !llvm.ptr<i64>
    %58 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %4, %57 : !llvm.ptr<i64>
    %59 = llvm.mlir.constant(1 : i64) : i64
    %60 = llvm.mlir.constant(0 : i64) : i64
    %61 = llvm.mlir.constant(2 : i64) : i64
    %62 = llvm.alloca %61 x i64 : (i64) -> !llvm.ptr<i64>
    %63 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %59, %62 : !llvm.ptr<i64>
    %64 = llvm.mlir.constant(1 : i64) : i64
    %65 = llvm.getelementptr %62[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %60, %65 : !llvm.ptr<i64>
    %66 = llvm.mlir.constant(2 : i64) : i64
    %67 = llvm.mlir.constant(1 : i64) : i64
    %68 = llvm.alloca %67 x i64 : (i64) -> !llvm.ptr<i64>
    %69 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %66, %68 : !llvm.ptr<i64>
    %70 = llvm.mlir.constant(1 : index) : i64
    %71 = llvm.mlir.constant(1 : i64) : i64
    %72 = llvm.alloca %71 x !llvm.ptr<ptr<i1>> : (i64) -> !llvm.ptr<ptr<ptr<i1>>>
    %73 = llvm.mlir.constant(0 : i64) : i64
    %74 = llvm.bitcast %36 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>> to !llvm.ptr<ptr<i1>>
    llvm.store %74, %72 : !llvm.ptr<ptr<ptr<i1>>>
    %75 = llvm.mlir.constant(1 : i64) : i64
    %76 = llvm.alloca %75 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %77 = llvm.mlir.constant(0 : i64) : i64
    %78 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %78, %76 : !llvm.ptr<ptr<i1>>
    %79 = llvm.mlir.constant(1 : index) : i64
    llvm.call @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(%76, %79, %38, %40, %52, %54, %57, %62, %68, %70, %72, %15) : (!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>) -> ()
    %80 = llvm.mlir.constant(0 : i64) : i64
    %81 = llvm.load %76 : !llvm.ptr<ptr<i1>>
    %82 = llvm.mlir.addressof @_vect2 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>>
    %83 = llvm.mlir.constant(6 : i64) : i64
    %84 = llvm.alloca %83 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i1>
    %85 = llvm.mlir.constant(6 : i64) : i64
    %86 = llvm.alloca %85 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %87 = llvm.mlir.constant(false) : i1
    %88 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %87, %84 : !llvm.ptr<i1>
    %89 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %19, %86 : !llvm.ptr<ptr<i1>>
    %90 = llvm.mlir.constant(false) : i1
    %91 = llvm.mlir.constant(1 : i64) : i64
    %92 = llvm.getelementptr %84[1] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %90, %92 : !llvm.ptr<i1>
    %93 = llvm.mlir.constant(1 : i64) : i64
    %94 = llvm.getelementptr %86[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    llvm.store %28, %94 : !llvm.ptr<ptr<i1>>
    %95 = llvm.mlir.constant(true) : i1
    %96 = llvm.mlir.constant(2 : i64) : i64
    %97 = llvm.getelementptr %84[2] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %95, %97 : !llvm.ptr<i1>
    %98 = llvm.mlir.constant(2 : i64) : i64
    %99 = llvm.getelementptr %86[2] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %100 = llvm.zext %2 : i1 to i64
    %101 = llvm.inttoptr %100 : i64 to !llvm.ptr<i1>
    llvm.store %101, %99 : !llvm.ptr<ptr<i1>>
    %102 = llvm.mlir.constant(true) : i1
    %103 = llvm.mlir.constant(3 : i64) : i64
    %104 = llvm.getelementptr %84[3] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %102, %104 : !llvm.ptr<i1>
    %105 = llvm.mlir.constant(3 : i64) : i64
    %106 = llvm.getelementptr %86[3] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %107 = llvm.zext %3 : i1 to i64
    %108 = llvm.inttoptr %107 : i64 to !llvm.ptr<i1>
    llvm.store %108, %106 : !llvm.ptr<ptr<i1>>
    %109 = llvm.mlir.constant(true) : i1
    %110 = llvm.mlir.constant(4 : i64) : i64
    %111 = llvm.getelementptr %84[4] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %109, %111 : !llvm.ptr<i1>
    %112 = llvm.mlir.constant(4 : i64) : i64
    %113 = llvm.getelementptr %86[4] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %114 = llvm.fpext %9 : f64 to f64
    %115 = llvm.bitcast %114 : f64 to i64
    %116 = llvm.inttoptr %115 : i64 to !llvm.ptr<i1>
    llvm.store %116, %113 : !llvm.ptr<ptr<i1>>
    %117 = llvm.mlir.constant(false) : i1
    %118 = llvm.mlir.constant(5 : i64) : i64
    %119 = llvm.getelementptr %84[5] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %117, %119 : !llvm.ptr<i1>
    %120 = llvm.mlir.constant(5 : i64) : i64
    %121 = llvm.getelementptr %86[5] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    llvm.store %81, %121 : !llvm.ptr<ptr<i1>>
    %122 = llvm.mlir.constant(6 : index) : i64
    %123 = llvm.mlir.constant(2 : i64) : i64
    %124 = llvm.alloca %123 x i64 : (i64) -> !llvm.ptr<i64>
    %125 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %4, %124 : !llvm.ptr<i64>
    %126 = llvm.mlir.constant(1 : i64) : i64
    %127 = llvm.getelementptr %124[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %8, %127 : !llvm.ptr<i64>
    %128 = llvm.mlir.constant(2 : i64) : i64
    %129 = llvm.alloca %128 x i64 : (i64) -> !llvm.ptr<i64>
    %130 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %4, %129 : !llvm.ptr<i64>
    %131 = llvm.mlir.constant(1 : i64) : i64
    %132 = llvm.getelementptr %129[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %4, %132 : !llvm.ptr<i64>
    %133 = llvm.mlir.constant(1 : i64) : i64
    %134 = llvm.mlir.constant(0 : i64) : i64
    %135 = llvm.mlir.constant(0 : i64) : i64
    %136 = llvm.mlir.constant(0 : i64) : i64
    %137 = llvm.mlir.constant(0 : i64) : i64
    %138 = llvm.mlir.constant(1 : i64) : i64
    %139 = llvm.mlir.constant(6 : i64) : i64
    %140 = llvm.alloca %139 x i64 : (i64) -> !llvm.ptr<i64>
    %141 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %133, %140 : !llvm.ptr<i64>
    %142 = llvm.mlir.constant(1 : i64) : i64
    %143 = llvm.getelementptr %140[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %134, %143 : !llvm.ptr<i64>
    %144 = llvm.mlir.constant(2 : i64) : i64
    %145 = llvm.getelementptr %140[2] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %135, %145 : !llvm.ptr<i64>
    %146 = llvm.mlir.constant(3 : i64) : i64
    %147 = llvm.getelementptr %140[3] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %136, %147 : !llvm.ptr<i64>
    %148 = llvm.mlir.constant(4 : i64) : i64
    %149 = llvm.getelementptr %140[4] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %137, %149 : !llvm.ptr<i64>
    %150 = llvm.mlir.constant(5 : i64) : i64
    %151 = llvm.getelementptr %140[5] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %138, %151 : !llvm.ptr<i64>
    %152 = llvm.mlir.constant(2 : i64) : i64
    %153 = llvm.mlir.constant(3 : i64) : i64
    %154 = llvm.mlir.constant(2 : i64) : i64
    %155 = llvm.alloca %154 x i64 : (i64) -> !llvm.ptr<i64>
    %156 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %152, %155 : !llvm.ptr<i64>
    %157 = llvm.mlir.constant(1 : i64) : i64
    %158 = llvm.getelementptr %155[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %153, %158 : !llvm.ptr<i64>
    %159 = llvm.mlir.constant(1 : index) : i64
    %160 = llvm.mlir.constant(1 : i64) : i64
    %161 = llvm.alloca %160 x !llvm.ptr<ptr<i1>> : (i64) -> !llvm.ptr<ptr<ptr<i1>>>
    %162 = llvm.mlir.constant(0 : i64) : i64
    %163 = llvm.bitcast %82 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>> to !llvm.ptr<ptr<i1>>
    llvm.store %163, %161 : !llvm.ptr<ptr<ptr<i1>>>
    %164 = llvm.mlir.constant(2 : i64) : i64
    %165 = llvm.alloca %164 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %166 = llvm.mlir.constant(0 : i64) : i64
    %167 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %167, %165 : !llvm.ptr<ptr<i1>>
    %168 = llvm.mlir.constant(1 : i64) : i64
    %169 = llvm.getelementptr %165[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %170 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %170, %169 : !llvm.ptr<ptr<i1>>
    %171 = llvm.mlir.constant(2 : index) : i64
    llvm.call @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(%165, %171, %84, %86, %122, %124, %129, %140, %155, %159, %161, %15) : (!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>) -> ()
    %172 = llvm.mlir.constant(0 : i64) : i64
    %173 = llvm.load %165 : !llvm.ptr<ptr<i1>>
    %174 = llvm.mlir.constant(1 : i64) : i64
    %175 = llvm.getelementptr %165[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %176 = llvm.load %175 : !llvm.ptr<ptr<i1>>
    %177 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%81, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %178 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%28, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %179 = llvm.mlir.addressof @_vect3 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>>
    %180 = llvm.mlir.constant(1 : i64) : i64
    %181 = llvm.alloca %180 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i1>
    %182 = llvm.mlir.constant(1 : i64) : i64
    %183 = llvm.alloca %182 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %184 = llvm.mlir.constant(false) : i1
    %185 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %184, %181 : !llvm.ptr<i1>
    %186 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %176, %183 : !llvm.ptr<ptr<i1>>
    %187 = llvm.mlir.constant(1 : index) : i64
    %188 = llvm.mlir.constant(1 : i64) : i64
    %189 = llvm.alloca %188 x i64 : (i64) -> !llvm.ptr<i64>
    %190 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %4, %189 : !llvm.ptr<i64>
    %191 = llvm.mlir.constant(1 : i64) : i64
    %192 = llvm.alloca %191 x i64 : (i64) -> !llvm.ptr<i64>
    %193 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %8, %192 : !llvm.ptr<i64>
    %194 = llvm.mlir.constant(1 : i64) : i64
    %195 = llvm.mlir.constant(1 : i64) : i64
    %196 = llvm.alloca %195 x i64 : (i64) -> !llvm.ptr<i64>
    %197 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %194, %196 : !llvm.ptr<i64>
    %198 = llvm.mlir.constant(2 : i64) : i64
    %199 = llvm.mlir.constant(1 : i64) : i64
    %200 = llvm.alloca %199 x i64 : (i64) -> !llvm.ptr<i64>
    %201 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %198, %200 : !llvm.ptr<i64>
    %202 = llvm.mlir.constant(1 : index) : i64
    %203 = llvm.mlir.constant(1 : i64) : i64
    %204 = llvm.alloca %203 x !llvm.ptr<ptr<i1>> : (i64) -> !llvm.ptr<ptr<ptr<i1>>>
    %205 = llvm.mlir.constant(0 : i64) : i64
    %206 = llvm.bitcast %179 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>> to !llvm.ptr<ptr<i1>>
    llvm.store %206, %204 : !llvm.ptr<ptr<ptr<i1>>>
    %207 = llvm.mlir.constant(1 : i64) : i64
    %208 = llvm.alloca %207 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %209 = llvm.mlir.constant(0 : i64) : i64
    %210 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %210, %208 : !llvm.ptr<ptr<i1>>
    %211 = llvm.mlir.constant(1 : index) : i64
    llvm.call @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(%208, %211, %181, %183, %187, %189, %192, %196, %200, %202, %204, %15) : (!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>) -> ()
    %212 = llvm.mlir.constant(0 : i64) : i64
    %213 = llvm.load %208 : !llvm.ptr<ptr<i1>>
    %214 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%176, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %215 = llvm.mlir.addressof @_vect4 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>>
    %216 = llvm.mlir.constant(4 : i64) : i64
    %217 = llvm.alloca %216 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i1>
    %218 = llvm.mlir.constant(4 : i64) : i64
    %219 = llvm.alloca %218 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %220 = llvm.mlir.constant(false) : i1
    %221 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %220, %217 : !llvm.ptr<i1>
    %222 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %173, %219 : !llvm.ptr<ptr<i1>>
    %223 = llvm.mlir.constant(false) : i1
    %224 = llvm.mlir.constant(1 : i64) : i64
    %225 = llvm.getelementptr %217[1] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %223, %225 : !llvm.ptr<i1>
    %226 = llvm.mlir.constant(1 : i64) : i64
    %227 = llvm.getelementptr %219[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    llvm.store %19, %227 : !llvm.ptr<ptr<i1>>
    %228 = llvm.mlir.constant(true) : i1
    %229 = llvm.mlir.constant(2 : i64) : i64
    %230 = llvm.getelementptr %217[2] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %228, %230 : !llvm.ptr<i1>
    %231 = llvm.mlir.constant(2 : i64) : i64
    %232 = llvm.getelementptr %219[2] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %233 = llvm.zext %2 : i1 to i64
    %234 = llvm.inttoptr %233 : i64 to !llvm.ptr<i1>
    llvm.store %234, %232 : !llvm.ptr<ptr<i1>>
    %235 = llvm.mlir.constant(false) : i1
    %236 = llvm.mlir.constant(3 : i64) : i64
    %237 = llvm.getelementptr %217[3] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %235, %237 : !llvm.ptr<i1>
    %238 = llvm.mlir.constant(3 : i64) : i64
    %239 = llvm.getelementptr %219[3] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    llvm.store %213, %239 : !llvm.ptr<ptr<i1>>
    %240 = llvm.mlir.constant(4 : index) : i64
    %241 = llvm.mlir.constant(1 : i64) : i64
    %242 = llvm.alloca %241 x i64 : (i64) -> !llvm.ptr<i64>
    %243 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %4, %242 : !llvm.ptr<i64>
    %244 = llvm.mlir.constant(1 : i64) : i64
    %245 = llvm.alloca %244 x i64 : (i64) -> !llvm.ptr<i64>
    %246 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %4, %245 : !llvm.ptr<i64>
    %247 = llvm.mlir.constant(1 : i64) : i64
    %248 = llvm.mlir.constant(0 : i64) : i64
    %249 = llvm.mlir.constant(0 : i64) : i64
    %250 = llvm.mlir.constant(1 : i64) : i64
    %251 = llvm.mlir.constant(4 : i64) : i64
    %252 = llvm.alloca %251 x i64 : (i64) -> !llvm.ptr<i64>
    %253 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %247, %252 : !llvm.ptr<i64>
    %254 = llvm.mlir.constant(1 : i64) : i64
    %255 = llvm.getelementptr %252[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %248, %255 : !llvm.ptr<i64>
    %256 = llvm.mlir.constant(2 : i64) : i64
    %257 = llvm.getelementptr %252[2] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %249, %257 : !llvm.ptr<i64>
    %258 = llvm.mlir.constant(3 : i64) : i64
    %259 = llvm.getelementptr %252[3] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %250, %259 : !llvm.ptr<i64>
    %260 = llvm.mlir.constant(1 : i64) : i64
    %261 = llvm.mlir.constant(1 : i64) : i64
    %262 = llvm.alloca %261 x i64 : (i64) -> !llvm.ptr<i64>
    %263 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %260, %262 : !llvm.ptr<i64>
    %264 = llvm.mlir.constant(1 : index) : i64
    %265 = llvm.mlir.constant(1 : i64) : i64
    %266 = llvm.alloca %265 x !llvm.ptr<ptr<i1>> : (i64) -> !llvm.ptr<ptr<ptr<i1>>>
    %267 = llvm.mlir.constant(0 : i64) : i64
    %268 = llvm.bitcast %215 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>> to !llvm.ptr<ptr<i1>>
    llvm.store %268, %266 : !llvm.ptr<ptr<ptr<i1>>>
    %269 = llvm.mlir.constant(1 : i64) : i64
    %270 = llvm.alloca %269 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %271 = llvm.mlir.constant(0 : i64) : i64
    %272 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %272, %270 : !llvm.ptr<ptr<i1>>
    %273 = llvm.mlir.constant(1 : index) : i64
    llvm.call @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(%270, %273, %217, %219, %240, %242, %245, %252, %262, %264, %266, %15) : (!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>) -> ()
    %274 = llvm.mlir.constant(0 : i64) : i64
    %275 = llvm.load %270 : !llvm.ptr<ptr<i1>>
    %276 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%213, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %277 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%173, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %278 = llvm.add %27, %8  : i64
    llvm.br ^bb1(%278, %275 : i64, !llvm.ptr<i1>)
  ^bb3:  // pred: ^bb1
    %279 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%23, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %280 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%19, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %281 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%28, %3, %2, %15) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %282 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%28, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %283 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%15) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}
