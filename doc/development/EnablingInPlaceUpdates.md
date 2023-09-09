# Enabling updating in place of data objects in kernels

This document outlines the steps necessary for Daphne/Kernel developers to enable updating objects in place in operation.

In-place updating refers to reusing and overwriting an input data object for the output. This eliminates the need to allocate a new data object for each operation, potentially reducing peak memory consumption and execution times.

**For example:**

```
X = readMatrix("X.csv");
Y = readMatrix("Y.csv");

Z = sqrt(X + Y);
```

Here, for adding two matrices elementwise, it results in the allocation of a new matrix with size of X and the same for calculating element-wise the square root of the intermediate result of X + Y. If X or Y is not used later in the Daphne application's execution, we could think about using either of them for storing the result. Thus avoiding the need for additional memory allocation.

**Using update in place in DAPHNE:**

If you want to use the in-place feature, you just need to add --update-in-place to the arguments for calling the DAPHNE. This leads to the execution of an additional compiler pass that enables the update in place. Appropriate JSON configuration can be used.

## Background

There are some condition that needs to hold true so that an update in place is possible:

### No interdependence of object elements while calculating a result

This is highly dependent on the operation/kernel. For instance, if the operation is performed element-wise (e.g. adding the first element of one matrix to the first element of another matrix), we can store the result directly in the position of the first element of the original matrix. Storing the result directly in the position of the first element of the original matrix is not feasible when the output calculation depends on other elements. For operations like matrix multiplication or convolution (with overlapping kernels), multiple elements need to be accessed and used to compute an element output. Overwriting the old values with the newly calculated one could lead to an incorrect result.

Therefore, only kernels that support this semantic can be updated in-place. Special algorithms may need to be implemented to make this possible (e.g. In-Place Transposition with cycles).

### No Future Use of data object/buffer in the application

Values can only be overridden if they are not used later in the application's execution. Typically, this is possible when an operand is not used as an operand in a succeding operation. However, due to the use of data buffers, views, and frames, the underlying values may be used multiple times in different data objects. Detecting this at compile-time is not trivial, so we rely on both compile-time and runtime analysis.

**Compile-time analysis**

At compile-time, we have the

While compiling the DAPHNE application to LLVM Instruction, in the Pass `FlagUpdateInPlace.cpp` we check whether a function can be used

In `RewriteToCallKernelOpPass.cpp`, the

**Runtime analysis**

Inside the individual kernels, we need to apply specific checks for identifiying the possiblity of in-placable operands.
In general, this is the case if the data object and the underlying data buffer is no

## Steps for enabling update in place for a kernel

To enable the updating in place for a kernel, we need to employ different steps. We expect that the kernel is already implemented according to [ImplementBuiltinKernel.md](doc/development/ImplementBuiltinKernel.md).

1. Mark the operation as InPlaceable
2. Change the corresponding shared libary function signature
3. Change the function signature of the kernel and specialization templates
4. Adapt the knowledge for improving execution

### 1. Mark the operation as InPlaceable

We need to add an MLIR op interface to the operation. This is necessary for the ﻿FlagUpdateInPlace.cpp pass to consider using the operands of the specific operation in place.

This is accompanied by changing the traits of an operation in DaphneOps.td by adding `DeclareOpInterfaceMethods<InPlaceOpInterface>` to the list. For instance:

```cpp
class Daphne_EwBinaryOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeFromArgs,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    ShapeEwBinary,
    CastArgsToResType,
    DeclareOpInterfaceMethods<InPlaceOpInterface>
])>
```

Additionally, we need to indicate which operand should be enabled for in-place updating. To achieve this, we must implement a ﻿getInPlaceOperands method that returns a vector of integers representing the positions of the operands to be considered for in-place update. This is done in `DaphneUpdateInPlaceOpInterface.cpp`.

For then Binary

### 2. Change shared library signature

### 3. Change

## Special Notes

Some examples are found here:
