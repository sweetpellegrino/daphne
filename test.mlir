module {
  func.func @main() {
    %0 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64
    %2 = "daphne.constant"() {value = 1.000000e-03 : f64} : () -> f64
    %3 = "daphne.constant"() {value = false} : () -> i1
    %4 = "daphne.constant"() {value = true} : () -> i1
    %5 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %6 = "daphne.constant"() {value = 0 : index} : () -> index
    %7 = "daphne.constant"() {value = 19 : index} : () -> index
    %8 = "daphne.constant"() {value = 1 : index} : () -> index
    %9 = "daphne.constant"() {value = 100 : index} : () -> index
    %10 = "daphne.constant"() {value = 20 : index} : () -> index
    %11 = "daphne.constant"() {value = 140731216678872 : ui64} : () -> ui64
    %12 = "daphne.createDaphneContext"(%11) : (ui64) -> !daphne.DaphneContext
    %13 = "daphne.randMatrix"(%9, %10, %1, %5, %5, %0) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<100x20xf64:sp[1.000000e+00]>
    %14 = "daphne.sliceCol"(%13, %6, %7) : (!daphne.Matrix<100x20xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<100x19xf64>
    %15 = "daphne.sliceCol"(%13, %7, %10) : (!daphne.Matrix<100x20xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<100x1xf64>
    %16 = "daphne.meanCol"(%14) : (!daphne.Matrix<100x19xf64>) -> !daphne.Matrix<1x19xf64>
    %17 = "daphne.ewSub"(%14, %16) {inPlaceFutureUse = [true, false]} : (!daphne.Matrix<100x19xf64>, !daphne.Matrix<1x19xf64>) -> !daphne.Matrix<100x19xf64>
    %18 = "daphne.stddevCol"(%14) : (!daphne.Matrix<100x19xf64>) -> !daphne.Matrix<1x19xf64>
    %19 = "daphne.ewDiv"(%17, %18) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<100x19xf64>, !daphne.Matrix<1x19xf64>) -> !daphne.Matrix<100x19xf64>
    %20 = "daphne.fill"(%5, %9, %8) : (f64, index, index) -> !daphne.Matrix<100x1xf64>
    %21 = "daphne.colBind"(%19, %20) : (!daphne.Matrix<100x19xf64>, !daphne.Matrix<100x1xf64>) -> !daphne.Matrix<100x20xf64>
    %22 = "daphne.fill"(%2, %10, %8) : (f64, index, index) -> !daphne.Matrix<20x1xf64>
    %23 = "daphne.syrk"(%21) : (!daphne.Matrix<100x20xf64>) -> !daphne.Matrix<20x20xf64>
    %24 = "daphne.diagMatrix"(%22) : (!daphne.Matrix<20x1xf64>) -> !daphne.Matrix<20x20xf64:sp[5.000000e-02]>
    %25 = "daphne.ewAdd"(%23, %24) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<20x20xf64>, !daphne.Matrix<20x20xf64:sp[5.000000e-02]>) -> !daphne.Matrix<20x20xf64>
    %26 = "daphne.gemv"(%21, %15) : (!daphne.Matrix<100x20xf64>, !daphne.Matrix<100x1xf64>) -> !daphne.Matrix<20x1xf64>
    %27 = "daphne.solve"(%25, %26) : (!daphne.Matrix<20x20xf64>, !daphne.Matrix<20x1xf64>) -> !daphne.Matrix<20x1xf64>
    "daphne.print"(%27, %4, %3) : (!daphne.Matrix<20x1xf64>, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}