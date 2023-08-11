//IR after llvm lowering
module {
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_ewSqrt__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_fill__DenseMatrix_double__double__size_t__size_t(!llvm.ptr<ptr<i1>>, f64, i64, i64, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(2 : index) : i64
    %4 = llvm.mlir.constant(140734524586976 : ui64) : i64
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
    llvm.call @_print__DenseMatrix_double__bool__bool(%12, %1, %2, %8) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.alloca %14 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %16 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %16, %15 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__double(%15, %12, %0, %8) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>) -> ()
    %17 = llvm.load %15 : !llvm.ptr<ptr<i1>>
    %18 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%12, %8) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %19 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%17, %1, %2, %8) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %20 = llvm.mlir.constant(1 : i64) : i64
    %21 = llvm.alloca %20 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %22 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %22, %21 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewSqrt__DenseMatrix_double__DenseMatrix_double(%21, %17, %8) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %23 = llvm.load %21 : !llvm.ptr<ptr<i1>>
    %24 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%17, %8) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %25 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%23, %1, %2, %8) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %26 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%23, %8) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %27 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%8) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}