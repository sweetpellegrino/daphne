module asm
  llvm.func @_destroyDaphneContext(!llvm.ptr<i1>)
  llvm.func @_print__char__bool__bool(!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_vect3(%arg0: !llvm.ptr<ptr<ptr<i1>>>, %arg1: !llvm.ptr<ptr<i1>>, %arg2: !llvm.ptr<i1>) {
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
    %15 = llvm.mlir.constant(1 : i64) : i64
    %16 = llvm.alloca %15 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %17 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %17, %16 : !llvm.ptr<ptr<i1>>
    llvm.call @_matMul__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double__bool__bool(%16, %1, %4, %9, %14, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %18 = llvm.load %16 : !llvm.ptr<ptr<i1>>
    %19 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%4, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %20 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%1, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %21 = llvm.mlir.constant(0 : i64) : i64
    %22 = llvm.load %arg0 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %18, %22 : !llvm.ptr<ptr<i1>>
    llvm.return
  }
  llvm.func @_transpose__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_vect2(%arg0: !llvm.ptr<ptr<ptr<i1>>>, %arg1: !llvm.ptr<ptr<i1>>, %arg2: !llvm.ptr<i1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.load %arg1 : !llvm.ptr<ptr<i1>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %4 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %4, %3 : !llvm.ptr<ptr<i1>>
    llvm.call @_transpose__DenseMatrix_double__DenseMatrix_double(%3, %1, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %5 = llvm.load %3 : !llvm.ptr<ptr<i1>>
    %6 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%1, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.load %arg0 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %5, %8 : !llvm.ptr<ptr<i1>>
    llvm.return
  }
  llvm.func @_decRef__Structure(!llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>)
  llvm.func @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>)
  llvm.func @_vect1(%arg0: !llvm.ptr<ptr<ptr<i1>>>, %arg1: !llvm.ptr<ptr<i1>>, %arg2: !llvm.ptr<i1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.load %arg1 : !llvm.ptr<ptr<i1>>
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %4 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %4, %3 : !llvm.ptr<ptr<i1>>
    llvm.call @_ewAdd__DenseMatrix_double__DenseMatrix_double__DenseMatrix_double(%3, %1, %1, %arg2) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %5 = llvm.load %3 : !llvm.ptr<ptr<i1>>
    %6 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%1, %arg2) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.load %arg0 : !llvm.ptr<ptr<ptr<i1>>>
    llvm.store %5, %8 : !llvm.ptr<ptr<i1>>
    llvm.return
  }
  llvm.func @_print__DenseMatrix_double__bool__bool(!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>)
  llvm.func @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>)
  llvm.func @_print__int64_t__bool__bool(i64, i1, i1, !llvm.ptr<i1>)
  llvm.func @_createDaphneContext__DaphneContext__uint64_t(!llvm.ptr<ptr<i1>>, i64)
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1.000000e+02 : f64) : f64
    %1 = llvm.mlir.constant(2.000000e+02 : f64) : f64
    %2 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(-1 : si64) : i64
    %4 = llvm.mlir.constant(13 : i64) : i64
    %5 = llvm.alloca %4 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.mlir.constant(72 : i8) : i8
    llvm.store %7, %5 : !llvm.ptr<i8>
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.getelementptr %5[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %10 = llvm.mlir.constant(101 : i8) : i8
    llvm.store %10, %9 : !llvm.ptr<i8>
    %11 = llvm.mlir.constant(2 : i64) : i64
    %12 = llvm.getelementptr %5[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %13 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %13, %12 : !llvm.ptr<i8>
    %14 = llvm.mlir.constant(3 : i64) : i64
    %15 = llvm.getelementptr %5[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %16 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %16, %15 : !llvm.ptr<i8>
    %17 = llvm.mlir.constant(4 : i64) : i64
    %18 = llvm.getelementptr %5[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %19 = llvm.mlir.constant(111 : i8) : i8
    llvm.store %19, %18 : !llvm.ptr<i8>
    %20 = llvm.mlir.constant(5 : i64) : i64
    %21 = llvm.getelementptr %5[5] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %22 = llvm.mlir.constant(32 : i8) : i8
    llvm.store %22, %21 : !llvm.ptr<i8>
    %23 = llvm.mlir.constant(6 : i64) : i64
    %24 = llvm.getelementptr %5[6] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %25 = llvm.mlir.constant(119 : i8) : i8
    llvm.store %25, %24 : !llvm.ptr<i8>
    %26 = llvm.mlir.constant(7 : i64) : i64
    %27 = llvm.getelementptr %5[7] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %28 = llvm.mlir.constant(111 : i8) : i8
    llvm.store %28, %27 : !llvm.ptr<i8>
    %29 = llvm.mlir.constant(8 : i64) : i64
    %30 = llvm.getelementptr %5[8] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %31 = llvm.mlir.constant(114 : i8) : i8
    llvm.store %31, %30 : !llvm.ptr<i8>
    %32 = llvm.mlir.constant(9 : i64) : i64
    %33 = llvm.getelementptr %5[9] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %34 = llvm.mlir.constant(108 : i8) : i8
    llvm.store %34, %33 : !llvm.ptr<i8>
    %35 = llvm.mlir.constant(10 : i64) : i64
    %36 = llvm.getelementptr %5[10] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %37 = llvm.mlir.constant(100 : i8) : i8
    llvm.store %37, %36 : !llvm.ptr<i8>
    %38 = llvm.mlir.constant(11 : i64) : i64
    %39 = llvm.getelementptr %5[11] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %40 = llvm.mlir.constant(33 : i8) : i8
    llvm.store %40, %39 : !llvm.ptr<i8>
    %41 = llvm.mlir.constant(12 : i64) : i64
    %42 = llvm.getelementptr %5[12] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %43 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %43, %42 : !llvm.ptr<i8>
    %44 = llvm.mlir.constant(5 : i64) : i64
    %45 = llvm.alloca %44 x i8 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i8>
    %46 = llvm.mlir.constant(0 : i64) : i64
    %47 = llvm.mlir.constant(66 : i8) : i8
    llvm.store %47, %45 : !llvm.ptr<i8>
    %48 = llvm.mlir.constant(1 : i64) : i64
    %49 = llvm.getelementptr %45[1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %50 = llvm.mlir.constant(121 : i8) : i8
    llvm.store %50, %49 : !llvm.ptr<i8>
    %51 = llvm.mlir.constant(2 : i64) : i64
    %52 = llvm.getelementptr %45[2] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %53 = llvm.mlir.constant(101 : i8) : i8
    llvm.store %53, %52 : !llvm.ptr<i8>
    %54 = llvm.mlir.constant(3 : i64) : i64
    %55 = llvm.getelementptr %45[3] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %56 = llvm.mlir.constant(33 : i8) : i8
    llvm.store %56, %55 : !llvm.ptr<i8>
    %57 = llvm.mlir.constant(4 : i64) : i64
    %58 = llvm.getelementptr %45[4] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %59 = llvm.mlir.constant(0 : i8) : i8
    llvm.store %59, %58 : !llvm.ptr<i8>
    %60 = llvm.mlir.constant(3 : si64) : i64
    %61 = llvm.mlir.constant(false) : i1
    %62 = llvm.mlir.constant(true) : i1
    %63 = llvm.mlir.constant(3 : index) : i64
    %64 = llvm.mlir.constant(2 : index) : i64
    %65 = llvm.mlir.constant(140732023337744 : ui64) : i64
    %66 = llvm.mlir.constant(1 : i64) : i64
    %67 = llvm.alloca %66 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %68 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %68, %67 : !llvm.ptr<ptr<i1>>
    llvm.call @_createDaphneContext__DaphneContext__uint64_t(%67, %65) : (!llvm.ptr<ptr<i1>>, i64) -> ()
    %69 = llvm.load %67 : !llvm.ptr<ptr<i1>>
    %70 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__int64_t__bool__bool(%60, %62, %61, %69) : (i64, i1, i1, !llvm.ptr<i1>) -> ()
    %71 = llvm.mlir.constant(1 : i64) : i64
    %72 = llvm.alloca %71 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %73 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %73, %72 : !llvm.ptr<ptr<i1>>
    llvm.call @_randMatrix__DenseMatrix_double__size_t__size_t__double__double__double__int64_t(%72, %64, %63, %0, %1, %2, %3, %69) : (!llvm.ptr<ptr<i1>>, i64, i64, f64, f64, f64, i64, !llvm.ptr<i1>) -> ()
    %74 = llvm.load %72 : !llvm.ptr<ptr<i1>>
    %75 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%74, %62, %61, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %76 = llvm.mlir.addressof @_vect1 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>>
    %77 = llvm.mlir.constant(1 : i64) : i64
    %78 = llvm.alloca %77 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i1>
    %79 = llvm.mlir.constant(1 : i64) : i64
    %80 = llvm.alloca %79 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %81 = llvm.mlir.constant(false) : i1
    %82 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %81, %78 : !llvm.ptr<i1>
    %83 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %74, %80 : !llvm.ptr<ptr<i1>>
    %84 = llvm.mlir.constant(1 : index) : i64
    %85 = llvm.mlir.constant(1 : i64) : i64
    %86 = llvm.alloca %85 x i64 : (i64) -> !llvm.ptr<i64>
    %87 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %64, %86 : !llvm.ptr<i64>
    %88 = llvm.mlir.constant(1 : i64) : i64
    %89 = llvm.alloca %88 x i64 : (i64) -> !llvm.ptr<i64>
    %90 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %63, %89 : !llvm.ptr<i64>
    %91 = llvm.mlir.constant(1 : i64) : i64
    %92 = llvm.mlir.constant(1 : i64) : i64
    %93 = llvm.alloca %92 x i64 : (i64) -> !llvm.ptr<i64>
    %94 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %91, %93 : !llvm.ptr<i64>
    %95 = llvm.mlir.constant(1 : i64) : i64
    %96 = llvm.mlir.constant(1 : i64) : i64
    %97 = llvm.alloca %96 x i64 : (i64) -> !llvm.ptr<i64>
    %98 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %95, %97 : !llvm.ptr<i64>
    %99 = llvm.mlir.constant(1 : index) : i64
    %100 = llvm.mlir.constant(1 : i64) : i64
    %101 = llvm.alloca %100 x !llvm.ptr<ptr<i1>> : (i64) -> !llvm.ptr<ptr<ptr<i1>>>
    %102 = llvm.mlir.constant(0 : i64) : i64
    %103 = llvm.bitcast %76 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>> to !llvm.ptr<ptr<i1>>
    llvm.store %103, %101 : !llvm.ptr<ptr<ptr<i1>>>
    %104 = llvm.mlir.constant(1 : i64) : i64
    %105 = llvm.alloca %104 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %106 = llvm.mlir.constant(0 : i64) : i64
    %107 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %107, %105 : !llvm.ptr<ptr<i1>>
    %108 = llvm.mlir.constant(1 : index) : i64
    llvm.call @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(%105, %108, %78, %80, %84, %86, %89, %93, %97, %99, %101, %69) : (!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>) -> ()
    %109 = llvm.mlir.constant(0 : i64) : i64
    %110 = llvm.load %105 : !llvm.ptr<ptr<i1>>
    %111 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%110, %62, %61, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %112 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%110, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %113 = llvm.mlir.addressof @_vect2 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>>
    %114 = llvm.mlir.constant(1 : i64) : i64
    %115 = llvm.alloca %114 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i1>
    %116 = llvm.mlir.constant(1 : i64) : i64
    %117 = llvm.alloca %116 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %118 = llvm.mlir.constant(false) : i1
    %119 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %118, %115 : !llvm.ptr<i1>
    %120 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %74, %117 : !llvm.ptr<ptr<i1>>
    %121 = llvm.mlir.constant(1 : index) : i64
    %122 = llvm.mlir.constant(1 : i64) : i64
    %123 = llvm.alloca %122 x i64 : (i64) -> !llvm.ptr<i64>
    %124 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %63, %123 : !llvm.ptr<i64>
    %125 = llvm.mlir.constant(1 : i64) : i64
    %126 = llvm.alloca %125 x i64 : (i64) -> !llvm.ptr<i64>
    %127 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %64, %126 : !llvm.ptr<i64>
    %128 = llvm.mlir.constant(1 : i64) : i64
    %129 = llvm.mlir.constant(1 : i64) : i64
    %130 = llvm.alloca %129 x i64 : (i64) -> !llvm.ptr<i64>
    %131 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %128, %130 : !llvm.ptr<i64>
    %132 = llvm.mlir.constant(2 : i64) : i64
    %133 = llvm.mlir.constant(1 : i64) : i64
    %134 = llvm.alloca %133 x i64 : (i64) -> !llvm.ptr<i64>
    %135 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %132, %134 : !llvm.ptr<i64>
    %136 = llvm.mlir.constant(1 : index) : i64
    %137 = llvm.mlir.constant(1 : i64) : i64
    %138 = llvm.alloca %137 x !llvm.ptr<ptr<i1>> : (i64) -> !llvm.ptr<ptr<ptr<i1>>>
    %139 = llvm.mlir.constant(0 : i64) : i64
    %140 = llvm.bitcast %113 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>> to !llvm.ptr<ptr<i1>>
    llvm.store %140, %138 : !llvm.ptr<ptr<ptr<i1>>>
    %141 = llvm.mlir.constant(1 : i64) : i64
    %142 = llvm.alloca %141 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %143 = llvm.mlir.constant(0 : i64) : i64
    %144 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %144, %142 : !llvm.ptr<ptr<i1>>
    %145 = llvm.mlir.constant(1 : index) : i64
    llvm.call @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(%142, %145, %115, %117, %121, %123, %126, %130, %134, %136, %138, %69) : (!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>) -> ()
    %146 = llvm.mlir.constant(0 : i64) : i64
    %147 = llvm.load %142 : !llvm.ptr<ptr<i1>>
    %148 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%147, %62, %61, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %149 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%147, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %150 = llvm.mlir.addressof @_vect3 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>>
    %151 = llvm.mlir.constant(4 : i64) : i64
    %152 = llvm.alloca %151 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr<i1>
    %153 = llvm.mlir.constant(4 : i64) : i64
    %154 = llvm.alloca %153 x !llvm.ptr<i1> {alignment = 1 : i64} : (i64) -> !llvm.ptr<ptr<i1>>
    %155 = llvm.mlir.constant(false) : i1
    %156 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %155, %152 : !llvm.ptr<i1>
    %157 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %74, %154 : !llvm.ptr<ptr<i1>>
    %158 = llvm.mlir.constant(false) : i1
    %159 = llvm.mlir.constant(1 : i64) : i64
    %160 = llvm.getelementptr %152[1] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %158, %160 : !llvm.ptr<i1>
    %161 = llvm.mlir.constant(1 : i64) : i64
    %162 = llvm.getelementptr %154[1] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    llvm.store %74, %162 : !llvm.ptr<ptr<i1>>
    %163 = llvm.mlir.constant(true) : i1
    %164 = llvm.mlir.constant(2 : i64) : i64
    %165 = llvm.getelementptr %152[2] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %163, %165 : !llvm.ptr<i1>
    %166 = llvm.mlir.constant(2 : i64) : i64
    %167 = llvm.getelementptr %154[2] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %168 = llvm.zext %61 : i1 to i64
    %169 = llvm.inttoptr %168 : i64 to !llvm.ptr<i1>
    llvm.store %169, %167 : !llvm.ptr<ptr<i1>>
    %170 = llvm.mlir.constant(true) : i1
    %171 = llvm.mlir.constant(3 : i64) : i64
    %172 = llvm.getelementptr %152[3] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
    llvm.store %170, %172 : !llvm.ptr<i1>
    %173 = llvm.mlir.constant(3 : i64) : i64
    %174 = llvm.getelementptr %154[3] : (!llvm.ptr<ptr<i1>>) -> !llvm.ptr<ptr<i1>>
    %175 = llvm.zext %62 : i1 to i64
    %176 = llvm.inttoptr %175 : i64 to !llvm.ptr<i1>
    llvm.store %176, %174 : !llvm.ptr<ptr<i1>>
    %177 = llvm.mlir.constant(4 : index) : i64
    %178 = llvm.mlir.constant(1 : i64) : i64
    %179 = llvm.alloca %178 x i64 : (i64) -> !llvm.ptr<i64>
    %180 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %64, %179 : !llvm.ptr<i64>
    %181 = llvm.mlir.constant(1 : i64) : i64
    %182 = llvm.alloca %181 x i64 : (i64) -> !llvm.ptr<i64>
    %183 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %64, %182 : !llvm.ptr<i64>
    %184 = llvm.mlir.constant(1 : i64) : i64
    %185 = llvm.mlir.constant(0 : i64) : i64
    %186 = llvm.mlir.constant(0 : i64) : i64
    %187 = llvm.mlir.constant(0 : i64) : i64
    %188 = llvm.mlir.constant(4 : i64) : i64
    %189 = llvm.alloca %188 x i64 : (i64) -> !llvm.ptr<i64>
    %190 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %184, %189 : !llvm.ptr<i64>
    %191 = llvm.mlir.constant(1 : i64) : i64
    %192 = llvm.getelementptr %189[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %185, %192 : !llvm.ptr<i64>
    %193 = llvm.mlir.constant(2 : i64) : i64
    %194 = llvm.getelementptr %189[2] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %186, %194 : !llvm.ptr<i64>
    %195 = llvm.mlir.constant(3 : i64) : i64
    %196 = llvm.getelementptr %189[3] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    llvm.store %187, %196 : !llvm.ptr<i64>
    %197 = llvm.mlir.constant(1 : i64) : i64
    %198 = llvm.mlir.constant(1 : i64) : i64
    %199 = llvm.alloca %198 x i64 : (i64) -> !llvm.ptr<i64>
    %200 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %197, %199 : !llvm.ptr<i64>
    %201 = llvm.mlir.constant(1 : index) : i64
    %202 = llvm.mlir.constant(1 : i64) : i64
    %203 = llvm.alloca %202 x !llvm.ptr<ptr<i1>> : (i64) -> !llvm.ptr<ptr<ptr<i1>>>
    %204 = llvm.mlir.constant(0 : i64) : i64
    %205 = llvm.bitcast %150 : !llvm.ptr<func<void (ptr<ptr<ptr<i1>>>, ptr<ptr<i1>>, ptr<i1>)>> to !llvm.ptr<ptr<i1>>
    llvm.store %205, %203 : !llvm.ptr<ptr<ptr<i1>>>
    %206 = llvm.mlir.constant(1 : i64) : i64
    %207 = llvm.alloca %206 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
    %208 = llvm.mlir.constant(0 : i64) : i64
    %209 = llvm.mlir.null : !llvm.ptr<i1>
    llvm.store %209, %207 : !llvm.ptr<ptr<i1>>
    %210 = llvm.mlir.constant(1 : index) : i64
    llvm.call @_vectorizedPipeline__DenseMatrix_double_variadic__size_t__bool__Structure_variadic__size_t__int64_t__int64_t__int64_t__int64_t__size_t__void_variadic(%207, %210, %152, %154, %177, %179, %182, %189, %199, %201, %203, %69) : (!llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i1>, !llvm.ptr<ptr<i1>>, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, !llvm.ptr<ptr<ptr<i1>>>, !llvm.ptr<i1>) -> ()
    %211 = llvm.mlir.constant(0 : i64) : i64
    %212 = llvm.load %207 : !llvm.ptr<ptr<i1>>
    %213 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%74, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %214 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__DenseMatrix_double__bool__bool(%212, %62, %61, %69) : (!llvm.ptr<i1>, i1, i1, !llvm.ptr<i1>) -> ()
    %215 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_decRef__Structure(%212, %69) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
    %216 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__char__bool__bool(%5, %62, %61, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
    %217 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_print__char__bool__bool(%45, %62, %61, %69) : (!llvm.ptr<i8>, i1, i1, !llvm.ptr<i1>) -> ()
    %218 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @_destroyDaphneContext(%69) : (!llvm.ptr<i1>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
}