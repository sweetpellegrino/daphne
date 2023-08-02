%8 = llvm.alloca %7 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
%10 = llvm.load %8 : !llvm.ptr<ptr<i1>>
llvm.call @_ewAdd__DenseMatrix_int64_t__DenseMatrix_int64_t__int64_t(%8, %10, %0, %6) : 
(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, i64, !llvm.ptr<i1>) -> ()

%14 = llvm.load %12 : !llvm.ptr<ptr<i1>>
%16 = llvm.mlir.constant(1 : i64) : i64
%17 = llvm.alloca %16 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
%18 = llvm.mlir.null : !llvm.ptr<i1>
llvm.store %18, %17 : !llvm.ptr<ptr<i1>>
llvm.call @_cast__DenseMatrix_double__DenseMatrix_int64_t(%17, %14, %6) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
%19 = llvm.load %17 : !llvm.ptr<ptr<i1>>
%20 = llvm.mlir.constant(1 : i64) : i64
llvm.call @_decRef__Structure(%14, %6) : (!llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
%21 = llvm.mlir.constant(1 : i64) : i64
%22 = llvm.alloca %21 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
%23 = llvm.mlir.null : !llvm.ptr<i1>
llvm.store %23, %22 : !llvm.ptr<ptr<i1>>
llvm.call @_ewSqrt__DenseMatrix_double__DenseMatrix_double(%22, %19, %6) : (!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, !llvm.ptr<i1>) -> ()
%24 = llvm.load %22 : !llvm.ptr<ptr<i1>>

%8 = llvm.alloca %7 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
%10 = llvm.load %8 : !llvm.ptr<ptr<i1>>
%12 = llvm.alloca %11 x !llvm.ptr<i1> : (i64) -> !llvm.ptr<ptr<i1>>
%13 = llvm.mlir.null : !llvm.ptr<i1>
llvm.store %13, %12 : !llvm.ptr<ptr<i1>>
llvm.call @_ewAdd__DenseMatrix_int64_t__DenseMatrix_int64_t__int64_t(%12, %10, %0, %6) : 
(!llvm.ptr<ptr<i1>>, !llvm.ptr<i1>, i64, !llvm.ptr<i1>) -> ()
%14 = llvm.load %12 : !llvm.ptr<ptr<i1>>