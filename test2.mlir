module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = false} : () -> i1
    %4 = "daphne.constant"() {value = 10 : index} : () -> index
    %5 = "daphne.constant"() {value = 1 : index} : () -> index
    %6 = "daphne.constant"() {value = 0 : index} : () -> index
    %7 = "daphne.constant"() {value = 140734155225080 : ui64} : () -> ui64
    %8 = "daphne.createDaphneContext"(%7) : (ui64) -> !daphne.DaphneContext
    %9 = "daphne.fill"(%0, %4, %4) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
    %10 = "daphne.fill"(%1, %4, %4) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
    %11 = "daphne.sliceRow"(%9, %6, %5) : (!daphne.Matrix<10x10xf64>, index, index) -> !daphne.Matrix<1x10xf64>
    %12 = "daphne.sliceCol"(%11, %6, %5) : (!daphne.Matrix<1x10xf64>, index, index) -> !daphne.Matrix<1x1xf64>
    %13 = "daphne.ewEq"(%12, %0) : (!daphne.Matrix<1x1xf64>, f64) -> !daphne.Matrix<1x1xf64>
    %14 = "daphne.cast"(%13) : (!daphne.Matrix<1x1xf64>) -> i1
    %15 = scf.if %14 -> (!daphne.Matrix<10x10xf64>) {
      %16 = "daphne.fill"(%0, %4, %4) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
      %17 = "daphne.sliceRow"(%9, %6, %5) : (!daphne.Matrix<10x10xf64>, index, index) -> !daphne.Matrix<1x10xf64>
      %18 = "daphne.sliceCol"(%17, %6, %5) : (!daphne.Matrix<1x10xf64>, index, index) -> !daphne.Matrix<1x1xf64>
      %19 = "daphne.ewEq"(%18, %0) : (!daphne.Matrix<1x1xf64>, f64) -> !daphne.Matrix<1x1xf64>
      %20 = "daphne.cast"(%19) : (!daphne.Matrix<1x1xf64>) -> i1
      %21 = scf.if %20 -> (!daphne.Matrix<10x10xf64>) {
        %22 = "daphne.ewAdd"(%9, %16) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
        scf.yield %22 : !daphne.Matrix<10x10xf64>
      } else {
        scf.yield %10 : !daphne.Matrix<10x10xf64>
      }
      scf.yield %21 : !daphne.Matrix<10x10xf64>
    } else {
      scf.yield %10 : !daphne.Matrix<10x10xf64>
    }
    "daphne.print"(%15, %2, %3) : (!daphne.Matrix<10x10xf64>, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}