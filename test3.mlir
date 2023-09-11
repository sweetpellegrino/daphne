IR after update in place flagging:
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 10 : index} : () -> index
    %5 = "daphne.constant"() {value = 5 : index} : () -> index
    %6 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %7 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 6 : index} : () -> index
    %9 = "daphne.constant"() {value = 1 : index} : () -> index
    %10 = "daphne.constant"() {value = -2.000000e+00 : f64} : () -> f64
    %11 = "daphne.constant"() {value = 2.000000e+00 : f64} : () -> f64
    %12 = "daphne.constant"() {value = 140730598036984 : ui64} : () -> ui64
    %13 = "daphne.createDaphneContext"(%12) : (ui64) -> !daphne.DaphneContext
    %14 = "daphne.randMatrix"(%4, %8, %1, %6, %6, %0) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<10x6xf64:sp[1.000000e+00]>
    %15 = "daphne.randMatrix"(%5, %8, %1, %6, %6, %0) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<5x6xf64:sp[1.000000e+00]>
    %16 = "daphne.cast"(%15) : (!daphne.Matrix<5x6xf64:sp[1.000000e+00]>) -> !daphne.Matrix<5x6xf64>
    %17 = scf.for %arg0 = %9 to %8 step %9 iter_args(%arg1 = %16) -> (!daphne.Matrix<5x6xf64>) {
      %18 = "daphne.cast"(%arg0) : (index) -> si64
      %19 = "daphne.ewMul"(%18, %7) {inPlaceFutureUse = [true, true]} : (si64, si64) -> si64
      %20 = "daphne.vectorizedPipeline"(%arg1, %11, %9, %5) ({
      ^bb0(%arg2: !daphne.Matrix<?x6xf64>, %arg3: f64):
        %24 = "daphne.ewPow"(%arg2, %arg3) {inPlaceFutureUse = [true, true]} : (!daphne.Matrix<?x6xf64>, f64) -> !daphne.Matrix<?x?xf64>
        %25 = "daphne.sumRow"(%24) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %26 = "daphne.transpose"(%25) {inPlaceFutureUse = [true]} : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        "daphne.return"(%26) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2], operand_segment_sizes = array<i32: 2, 1, 1, 0>, splits = [1, 0]} : (!daphne.Matrix<5x6xf64>, f64, index, index) -> !daphne.Matrix<1x5xf64>
      %21:2 = "daphne.vectorizedPipeline"(%14, %arg1, %2, %3, %10, %20, %5, %9, %4, %5) ({
      ^bb0(%arg2: !daphne.Matrix<?x6xf64:sp[1.000000e+00]>, %arg3: !daphne.Matrix<5x6xf64>, %arg4: i1, %arg5: i1, %arg6: f64, %arg7: !daphne.Matrix<?x5xf64>):
        %24 = "daphne.matMul"(%arg2, %arg3, %arg4, %arg5) : (!daphne.Matrix<?x6xf64:sp[1.000000e+00]>, !daphne.Matrix<5x6xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
        %25 = "daphne.ewMul"(%24, %arg6) {inPlaceFutureUse = [true, true]} : (!daphne.Matrix<?x?xf64>, f64) -> !daphne.Matrix<?x?xf64>
        %26 = "daphne.ewAdd"(%25, %arg7) {inPlaceFutureUse = [true, true]} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x5xf64>) -> !daphne.Matrix<?x?xf64>
        %27 = "daphne.minRow"(%26) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %28 = "daphne.ewLe"(%26, %27) {inPlaceFutureUse = [true, true]} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %29 = "daphne.sumRow"(%28) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %30 = "daphne.ewDiv"(%28, %29) {inPlaceFutureUse = [true, true]} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %31 = "daphne.transpose"(%30) {inPlaceFutureUse = [true]} : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %32 = "daphne.sumCol"(%30) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        "daphne.return"(%31, %32) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2, 3], operand_segment_sizes = array<i32: 6, 2, 2, 0>, splits = [1, 0, 0, 0, 0, 1]} : (!daphne.Matrix<10x6xf64:sp[1.000000e+00]>, !daphne.Matrix<5x6xf64>, i1, i1, f64, !daphne.Matrix<1x5xf64>, index, index, index, index) -> (!daphne.Matrix<5x10xf64>, !daphne.Matrix<1x5xf64>)
      %22 = "daphne.vectorizedPipeline"(%21#1, %5, %9) ({
      ^bb0(%arg2: !daphne.Matrix<?x5xf64>):
        %24 = "daphne.transpose"(%arg2) {inPlaceFutureUse = [true]} : (!daphne.Matrix<?x5xf64>) -> !daphne.Matrix<?x?xf64>
        "daphne.return"(%24) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [2], operand_segment_sizes = array<i32: 1, 1, 1, 0>, splits = [1]} : (!daphne.Matrix<1x5xf64>, index, index) -> !daphne.Matrix<5x1xf64>
      %23 = "daphne.vectorizedPipeline"(%21#0, %14, %2, %22, %5, %8) ({
      ^bb0(%arg2: !daphne.Matrix<?x10xf64>, %arg3: !daphne.Matrix<10x6xf64:sp[1.000000e+00]>, %arg4: i1, %arg5: !daphne.Matrix<?x1xf64>):
        %24 = "daphne.matMul"(%arg2, %arg3, %arg4, %arg4) : (!daphne.Matrix<?x10xf64>, !daphne.Matrix<10x6xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<?x?xf64>
        %25 = "daphne.ewDiv"(%24, %arg5) {inPlaceFutureUse = [true, true]} : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64>
        "daphne.return"(%25) : (!daphne.Matrix<?x?xf64>) -> ()
      }, {
      }) {combines = [1], operand_segment_sizes = array<i32: 4, 1, 1, 0>, splits = [1, 0, 0, 1]} : (!daphne.Matrix<5x10xf64>, !daphne.Matrix<10x6xf64:sp[1.000000e+00]>, i1, !daphne.Matrix<5x1xf64>, index, index) -> !daphne.Matrix<5x6xf64>
      scf.yield %23 : !daphne.Matrix<5x6xf64>
    }
    "daphne.print"(%17, %3, %2) : (!daphne.Matrix<5x6xf64>, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}