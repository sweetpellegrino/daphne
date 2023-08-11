module {
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_print__DenseMatrix_int64_t__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_ewSqrt__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_incRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>)
  llvm.func @_cast__DenseMatrix_double__DenseMatrix_int64_t(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_print__Frame__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_createFrame__Frame__Structure_variadic__size_t__char_variadic__size_t(!llvm.ptr<ptr<i1>>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<ptr<i8>>, i64, !llvm.ptr<i1>)
  llvm.func @_randMatrix__DenseMatrix_int64_t__size_t__size_t__int64_t__int64_t__double__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, i64, i64, f64, i64, !llvm.ptr<i1>)
  llvm.func @_seq__DenseMatrix_int64_t__int64_t__int64_t__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, i64, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : si64) : i64
    %1 = llvm.mlir.constant(10 : si64) : i64
    %2 = llvm.mlir.constant(0 : si64) : i64
    %3 = llvm.mlir.constant(100 : si64) : i64
    %4 = llvm.mlir.constant(-1 : si64) : i64
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.alloca %5 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.mlir.constant(97 : i8) : i8
    llvm.store %8, %6 : !llvm.ptr<i8>
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.getelementptr %6[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %11 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %11, %10 : !llvm.ptr<i8>
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.alloca %12 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.mlir.constant(98 : i8) : i8
    llvm.store %15, %13 : !llvm.ptr<i8>
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.getelementptr %13[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %18 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %18, %17 : !llvm.ptr<i8>
    %19 = llvm.mlir.constant(7 : si64) : i64
    %20 = llvm.mlir.constant(3 : si64) : i64
    %21 = llvm.mlir.constant(200 : si64) : i64
    %22 = llvm.mlir.constant(2 : i64) : i64
    %23 = llvm.alloca %22 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %24 = llvm.mlir.constant(0 : i64) : i64
    %25 = llvm.mlir.constant(100 : i8) : i8
    llvm.store %25, %23 : !llvm.ptr<i8>
    %26 = llvm.mlir.constant(1 : i64) : i64
    %27 = llvm.getelementptr %23[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %28 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %28, %27 : !llvm.ptr<i8>
    %29 = llvm.mlir.constant(2 : i64) : i64
    %30 = llvm.alloca %29 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %31 = llvm.mlir.constant(0 : i64) : i64
    %32 = llvm.mlir.constant(99 : i8) : i8
    llvm.store %32, %30 : !llvm.ptr<i8>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %30[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %35 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %35, %34 : !llvm.ptr<i8>
    %36 = llvm.mlir.constant(true) : i1
    %37 = llvm.mlir.constant(false) : i1
    %38 = llvm.mlir.constant(11 : index) : i64
    %39 = llvm.mlir.constant(3 : index) : i64
    %40 = llvm.mlir.constant(1 : index) : i64
    %41 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %42 = llvm.mlir.constant(140729419817728 : ui64) : i64
    %43 = llvm.mlir.constant(1 : i64) : i64
    %44 = llvm.alloca %43 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %45 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %45, %44 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%44, %42) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %46 = llvm.load %44 : !llvm.ptr<ptr<i1>>
    %47 = llvm.mlir.constant(1 : i64) : i64
    %48 = llvm.alloca %47 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %49 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %49, %48 : !llvm.ptr<ptr<i1>>
    llvm.call @_seq__DenseMatrix_int64_t__int64_t__int64_t__int64_t(%48, %2, %1, %0, %46) : (!llvm.ptr<ptr<i1>>, i64, i64, i64, !llvm.ptr<i1>) -> ()
    %50 = llvm.load %48 : !llvm.ptr<ptr<i1>>
    %51 = llvm.mlir.constant(1 : i64) : i64
    %52 = llvm.alloca %51 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %53 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %53, %52 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_int64_t__size_t__size_t__int64_t__int64_t__double__int64_t(%52, %38, %40, %2, %3, %41, %4, %46) : (!llvm.ptr<ptr<i1>>, i64, i64, i64, i64, f64, i64, !llvm.ptr<i1>) -> ()
    %54 = llvm.load %52 : !llvm.ptr<ptr<i1>>
    %55 = llvm.mlir.constant(2 : i64) : i64
    %56 = llvm.alloca %55 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %57 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %50, %56 : !llvm.ptr<ptr<i1>>
    %58 = llvm.mlir.constant(1 : i64) : i64
    %59 = llvm.getelementptr %56[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    llvm.store %54, %59 : !llvm.ptr<ptr<i1>>
    %60 = llvm.mlir.constant(2 : index) : i64
    %61 = llvm.mlir.constant(2 : i64) : i64
    %62 = llvm.alloca %61 x !llvm.ptr<i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i8>>
    %63 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %6, %62 : !llvm.ptr<ptr<i8>>
    %64 = llvm.mlir.constant(1 : i64) : i64
    %65 = llvm.getelementptr %62[1] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    llvm.store %13, %65 : !llvm.ptr<ptr<i8>>
    %66 = llvm.mlir.constant(2 : index) : i64
    %67 = llvm.mlir.constant(1 : i64) : i64
    %68 = llvm.alloca %67 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %69 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %69, %68 : !llvm.ptr<ptr<i1>>
    llvm.call @_createFrame__Frame__Structure_variadic__size_t__char_variadic__size_t(%68, %56, %60, %62, %66, %46) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<ptr<i8>>, i64, !llvm.ptr<i1>) -> ()
    %70 = llvm.load %68 : !llvm.ptr<ptr<i1>>
    %71 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%70, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %72 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%54, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %73 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%50, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %74 = llvm.mlir.constant(1 : i64) : i64
    %75 = llvm.alloca %74 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %76 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %76, %75 : !llvm.ptr<ptr<i1>>
    llvm.call @_seq__DenseMatrix_int64_t__int64_t__int64_t__int64_t(%75, %2, %19, %20, %46) : (!llvm.ptr<ptr<i1>>, i64, i64, i64, !llvm.ptr<i1>) -> ()
    %77 = llvm.load %75 : !llvm.ptr<ptr<i1>>
    %78 = llvm.mlir.constant(1 : i64) : i64
    %79 = llvm.alloca %78 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %80 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %80, %79 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_int64_t__size_t__size_t__int64_t__int64_t__double__int64_t(%79, %39, %40, %3, %21, %41, %4, %46) : (!llvm.ptr<ptr<i1>>, i64, i64, i64, i64, f64, i64, !llvm.ptr<i1>) -> ()
    %81 = llvm.load %79 : !llvm.ptr<ptr<i1>>
    %82 = llvm.mlir.constant(2 : i64) : i64
    %83 = llvm.alloca %82 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %84 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %77, %83 : !llvm.ptr<ptr<i1>>
    %85 = llvm.mlir.constant(1 : i64) : i64
    %86 = llvm.getelementptr %83[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    llvm.store %81, %86 : !llvm.ptr<ptr<i1>>
    %87 = llvm.mlir.constant(2 : index) : i64
    %88 = llvm.mlir.constant(2 : i64) : i64
    %89 = llvm.alloca %88 x !llvm.ptr<i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i8>>
    %90 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %23, %89 : !llvm.ptr<ptr<i8>>
    %91 = llvm.mlir.constant(1 : i64) : i64
    %92 = llvm.getelementptr %89[1] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    llvm.store %30, %92 : !llvm.ptr<ptr<i8>>
    %93 = llvm.mlir.constant(2 : index) : i64
    %94 = llvm.mlir.constant(1 : i64) : i64
    %95 = llvm.alloca %94 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %96 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %96, %95 : !llvm.ptr<ptr<i1>>
    llvm.call @_createFrame__Frame__Structure_variadic__size_t__char_variadic__size_t(%95, %83, %87, %89, %93, %46) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<ptr<i8>>, i64, !llvm.ptr<i1>) -> ()
    %97 = llvm.load %95 : !llvm.ptr<ptr<i1>>
    %98 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%81, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %99 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__Frame__bool__bool(%97, %36, %37, %46) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %100 = llvm.mlir.constant(1 : i64) : i64
    %101 = llvm.alloca %100 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %102 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %102, %101 : !llvm.ptr<ptr<i1>>
    llvm.call @_cast__DenseMatrix_double__DenseMatrix_int64_t(%101, %77, %46) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %103 = llvm.load %101 : !llvm.ptr<ptr<i1>>
    %104 = llvm.mlir.constant(1 : i64) : i64
    %105 = llvm.alloca %104 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %106 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %106, %105 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__double(%105, %103, %41, %46) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, f64, !llvm.ptr<i1>) -> ()
    %107 = llvm.load %105 : !llvm.ptr<ptr<i1>>
    %108 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%103, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %109 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_incRef__Structure(%107, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    llvm.call @_ewSqrt__DenseMatrix_double__DenseMatrix_double(%105, %107, %46) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %110 = llvm.load %105 : !llvm.ptr<ptr<i1>>
    %111 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%107, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %112 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__Frame__bool__bool(%97, %36, %37, %46) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %113 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_incRef__Structure(%110, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    llvm.call @_ewSqrt__DenseMatrix_double__DenseMatrix_double(%105, %110, %46) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %114 = llvm.load %105 : !llvm.ptr<ptr<i1>>
    %115 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%110, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %116 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__Frame__bool__bool(%97, %36, %37, %46) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %117 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_int64_t__bool__bool(%77, %36, %37, %46) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %118 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%77, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %119 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%114, %36, %37, %46) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %120 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%114, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %121 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__Frame__bool__bool(%97, %36, %37, %46) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %122 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%97, %46) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %123 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%46) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}