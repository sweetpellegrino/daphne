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
    %11 = "daphne.constant"() {value = 140726049251504 : ui64} : () -> ui64
    %12 = "daphne.createDaphneContext"(%11) : (ui64) -> !daphne.DaphneContext
    %13 = "daphne.randMatrix"(%6, %6, %3, %7, %7, %2) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    %14 = "daphne.randMatrix"(%6, %6, %3, %7, %7, %2) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<5x5xf64:sp[1.000000e+00]>
    %15 = "daphne.cast"(%14) : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>) -> !daphne.Matrix<5x5xf64>
    %16 = scf.for %arg0 = %10 to %9 step %10 iter_args(%arg1 = %15) -> (!daphne.Matrix<5x5xf64>) {
      %17 = "daphne.cast"(%arg0) : (index) -> si64
      %18 = "daphne.ewMul"(%17, %8) {inPlaceFutureUse = [false, false]} : (si64, si64) -> si64
      %19 = "daphne.matMul"(%13, %arg1, %4, %5) : (!daphne.Matrix<5x5xf64:sp[1.000000e+00]>, !daphne.Matrix<5x5xf64>, i1, i1) -> !daphne.Matrix<5x5xf64>
      %20 = "daphne.ewMul"(%19, %1) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<5x5xf64>, f64) -> !daphne.Matrix<5x5xf64>
      %21 = "daphne.ewPow"(%arg1, %0) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<5x5xf64>, f64) -> !daphne.Matrix<5x5xf64>
      %22 = "daphne.sumRow"(%21) : (!daphne.Matrix<5x5xf64>) -> !daphne.Matrix<5x1xf64>
      %23 = "daphne.transpose"(%22) : (!daphne.Matrix<5x1xf64>) -> !daphne.Matrix<1x5xf64>
      %24 = "daphne.ewAdd"(%20, %23) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<1x5xf64>) -> !daphne.Matrix<5x5xf64>
      %25 = "daphne.minRow"(%24) : (!daphne.Matrix<5x5xf64>) -> !daphne.Matrix<5x1xf64>
      %26 = "daphne.ewLe"(%24, %25) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x1xf64>) -> !daphne.Matrix<5x5xf64>
      %27 = "daphne.sumRow"(%26) : (!daphne.Matrix<5x5xf64>) -> !daphne.Matrix<5x1xf64>
      %28 = "daphne.ewDiv"(%26, %27) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x1xf64>) -> !daphne.Matrix<5x5xf64>
      %29 = "daphne.sumCol"(%28) : (!daphne.Matrix<5x5xf64>) -> !daphne.Matrix<1x5xf64>
      %30 = "daphne.transpose"(%28) : (!daphne.Matrix<5x5xf64>) -> !daphne.Matrix<5x5xf64>
      %31 = "daphne.matMul"(%30, %13, %4, %4) : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x5xf64:sp[1.000000e+00]>, i1, i1) -> !daphne.Matrix<5x5xf64>
      %32 = "daphne.transpose"(%29) : (!daphne.Matrix<1x5xf64>) -> !daphne.Matrix<5x1xf64>
      %33 = "daphne.ewDiv"(%31, %32) {inPlaceFutureUse = [false, false]} : (!daphne.Matrix<5x5xf64>, !daphne.Matrix<5x1xf64>) -> !daphne.Matrix<5x5xf64>
      scf.yield %33 : !daphne.Matrix<5x5xf64>
    }
    "daphne.print"(%16, %5, %4) : (!daphne.Matrix<5x5xf64>, i1, i1) -> ()
    "daphne.destroyDaphneContext"() : () -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  llvm.func @_sumCol__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_minRow__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_transpose__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_sumRow__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewPow__DenseMatrix_double__DenseMatrix_double__double__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, i1, !llvm.ptr<i1>)
  llvm.func @_ewMul__DenseMatrix_double__DenseMatrix_double__double__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, i1, !llvm.ptr<i1>)
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
    %6 = llvm.mlir.constant(5 : index) : i64
    %7 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %8 = llvm.mlir.constant(1 : si64) : i64
    %9 = llvm.mlir.constant(6 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(140733161976512 : ui64) : i64
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
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%17, %6, %6, %3, %7, %7, %2, %15) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %19 = llvm.load %17 : !llvm.ptr<ptr<i1>>
    %20 = llvm.mlir.constant(1 : i64) : i64
    %21 = llvm.alloca %20 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %22 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %22, %21 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%21, %6, %6, %3, %7, %7, %2, %15) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %23 = llvm.load %21 : !llvm.ptr<ptr<i1>>
    %24 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_incRef__Structure(%23, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %25 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%23, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %26 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_incRef__Structure(%23, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    llvm.br ^bb1(%10, %23 : i64, !llvm.ptr<i1>)
  ^bb1(%27: i64, %28: !llvm.ptr<i1>):  // 2 preds: ^bb0, ^bb2
    %29 = llvm.icmp "slt" %27, %9 : i64
    llvm.cond_br %29, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %30 = llvm.mlir.constant(1 : i64) : i64
    %31 = llvm.alloca %30 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.call @_cast__int64_t__size_t(%31, %27, %15) : (!llvm.ptr<i64>, i64, !llvm.ptr<i1>) -> ()
    %32 = llvm.load %31 : !llvm.ptr<i64>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.alloca %33 x i64 : (i64) -> !llvm.ptr<i64>
    llvm.call @_ewMul__int64_t__int64_t__int64_t(%34, %32, %8, %15) : (!llvm.ptr<i64>, i64, i64, !llvm.ptr<i1>) -> ()
    %35 = llvm.load %34 : !llvm.ptr<i64>
    %36 = llvm.mlir.constant(1 : i64) : i64
    %37 = llvm.alloca %36 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %38 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %38, %37 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%37, %19, %28, %4, %5, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %39 = llvm.load %37 : !llvm.ptr<ptr<i1>>
    %40 = llvm.mlir.constant(false) : i1
    %41 = llvm.mlir.constant(1 : i64) : i64
    %42 = llvm.alloca %41 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %43 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %43, %42 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewMul__DenseMatrix_double__DenseMatrix_double__double__bool(%42, %39, %1, %40, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, i1, !llvm.ptr<i1>) -> ()
    %44 = llvm.load %42 : !llvm.ptr<ptr<i1>>
    %45 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%39, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %46 = llvm.mlir.constant(false) : i1
    %47 = llvm.mlir.constant(1 : i64) : i64
    %48 = llvm.alloca %47 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %49 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %49, %48 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewPow__DenseMatrix_double__DenseMatrix_double__double__bool(%48, %28, %0, %46, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, i1, !llvm.ptr<i1>) -> ()
    %50 = llvm.load %48 : !llvm.ptr<ptr<i1>>
    %51 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%28, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %52 = llvm.mlir.constant(1 : i64) : i64
    %53 = llvm.alloca %52 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %54 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %54, %53 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumRow__DenseMatrix_double__DenseMatrix_double(%53, %50, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %55 = llvm.load %53 : !llvm.ptr<ptr<i1>>
    %56 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%50, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %57 = llvm.mlir.constant(1 : i64) : i64
    %58 = llvm.alloca %57 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %59 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %59, %58 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%58, %55, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %60 = llvm.load %58 : !llvm.ptr<ptr<i1>>
    %61 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%55, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %62 = llvm.mlir.constant(false) : i1
    %63 = llvm.mlir.constant(false) : i1
    %64 = llvm.mlir.constant(1 : i64) : i64
    %65 = llvm.alloca %64 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %66 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %66, %65 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%65, %44, %60, %62, %63, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %67 = llvm.load %65 : !llvm.ptr<ptr<i1>>
    %68 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%60, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %69 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%44, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %70 = llvm.mlir.constant(1 : i64) : i64
    %71 = llvm.alloca %70 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %72 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %72, %71 : !llvm.ptr<ptr<i1>>
    llvm.call @_minRow__DenseMatrix_double__DenseMatrix_double(%71, %67, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %73 = llvm.load %71 : !llvm.ptr<ptr<i1>>
    %74 = llvm.mlir.constant(false) : i1
    %75 = llvm.mlir.constant(false) : i1
    %76 = llvm.mlir.constant(1 : i64) : i64
    %77 = llvm.alloca %76 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %78 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %78, %77 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewLe__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%77, %67, %73, %74, %75, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %79 = llvm.load %77 : !llvm.ptr<ptr<i1>>
    %80 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%73, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %81 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%67, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %82 = llvm.mlir.constant(1 : i64) : i64
    %83 = llvm.alloca %82 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %84 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %84, %83 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumRow__DenseMatrix_double__DenseMatrix_double(%83, %79, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %85 = llvm.load %83 : !llvm.ptr<ptr<i1>>
    %86 = llvm.mlir.constant(false) : i1
    %87 = llvm.mlir.constant(false) : i1
    %88 = llvm.mlir.constant(1 : i64) : i64
    %89 = llvm.alloca %88 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %90 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %90, %89 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%89, %79, %85, %86, %87, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %91 = llvm.load %89 : !llvm.ptr<ptr<i1>>
    %92 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%85, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %93 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%79, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %94 = llvm.mlir.constant(1 : i64) : i64
    %95 = llvm.alloca %94 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %96 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %96, %95 : !llvm.ptr<ptr<i1>>
    llvm.call @_sumCol__DenseMatrix_double__DenseMatrix_double(%95, %91, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %97 = llvm.load %95 : !llvm.ptr<ptr<i1>>
    %98 = llvm.mlir.constant(1 : i64) : i64
    %99 = llvm.alloca %98 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %100 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %100, %99 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%99, %91, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %101 = llvm.load %99 : !llvm.ptr<ptr<i1>>
    %102 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%91, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %103 = llvm.mlir.constant(1 : i64) : i64
    %104 = llvm.alloca %103 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %105 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %105, %104 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%104, %101, %19, %4, %4, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %106 = llvm.load %104 : !llvm.ptr<ptr<i1>>
    %107 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%101, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %108 = llvm.mlir.constant(1 : i64) : i64
    %109 = llvm.alloca %108 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %110 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %110, %109 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%109, %97, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %111 = llvm.load %109 : !llvm.ptr<ptr<i1>>
    %112 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%97, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %113 = llvm.mlir.constant(false) : i1
    %114 = llvm.mlir.constant(false) : i1
    %115 = llvm.mlir.constant(1 : i64) : i64
    %116 = llvm.alloca %115 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %117 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %117, %116 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewDiv__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%116, %106, %111, %113, %114, %15) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %118 = llvm.load %116 : !llvm.ptr<ptr<i1>>
    %119 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%111, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %120 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%106, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %121 = llvm.add %27, %10  : i64
    llvm.br ^bb1(%121, %118 : i64, !llvm.ptr<i1>)
  ^bb3:  // pred: ^bb1
    %122 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%23, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %123 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%19, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %124 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%28, %5, %4, %15) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %125 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%28, %15) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %126 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%15) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}