**Motivation:** By default, DAPHNE maps logical operations from linear algebra and relational algebra to kernels (physical operators) written in C++. Essentially, a kernel is a C++ function which consumes and produces DAPHNE data objects (matrices and frames). Typically, a kernel always creates and allocates a new data object for its output, and never modifies its input data object(s). This approach is simple from the DAPHNE compiler and runtime point of view, and relies on DAPHNE’s garbage collection to automatically free data objects with no future uses.

Interestingly, there are many situations when in-place-updates, which re-use and overwrite an input data object for the output, could be applied. Such in-place updates can have a number of advantages:

* The peak memory consumption could be reduced. That way, larger data volumes could be processed in-memory (as opposed to stored on disk), and locally (as opposed to distributed to additional nodes).
* The cache-efficiency and, thereby, the performance of a kernel could be improved since the input and output data could reside in the same cache lines.
* The number of (de)allocations from the operating system could be reduced due to re-uses of the data objects and underlying buffers.

As an example, consider the very simple expression** **`sqrt(X + 1)`, where** **`X` is a (large) matrix. This expression consists of two operations: first, the value 1 is added to all elements in X, and then, the square root of all resulting elements is calculated. When done naively, both operations need to allocate a new matrix for their result. However, if** **`X` is used nowhere else later on, then** **`X + 1` could store its output into the input data object. Likewise,** **`sqrt(X + 1)` could overwrite its input. Consequently, this expression could be evaluated without allocating any additional data objects.

**Task:** This project is about enabling this kind of update-in-place optimization in DAPHNE and evaluating its impact on the runtime and memory consumption of DaphneDSL scripts. The solution will touch upon the DAPHNE compiler and runtime. The implementation is in C++.

**Hints on approaching this task:**

* Familiarize yourself with (1) exploring the definition-usage-chain of MLIR operations, and (2) memory management and garbage collection in DAPHNE (for both data objects and their underlying data buffers).
* Think of a design to enable update-in-place operations in DAPHNE. Which logical operations in DaphneIR qualify for update-in-place? Under what conditions can update-in-place be applied for a particular operation in the context of the surrounding IR and how can the system identify these situations automatically? What is required in the DAPHNE compiler, what in the runtime, and what in the individual kernels?
* Implement your design, including test cases and documentation.
* Think about meaningful experiments (micro benchmarks and end-to-end algorithms), which reveal the strengths (and potential weaknesses) of your approach. Think about which characteristics of the DaphneDSL script and used data as well as system parameters (e.g., (not) using DAPHNE’s vectorized engine) could have an impact on your approach. Conduct the experiments, visualize, and interpret the results.

**Side notes:**

* The solution should not require the user to specify which operations should be update-in-place, it should work completely automatically.
* It would be great if the update-in-place feature could be turned on/off by means of the DAPHNE user configuration and command-line arguments.
* For now, you can focus on operations on dense matrices.

**Suggested task extensions for larger teams:**

* Enable update-in-place operations on sparse matrices (e.g.,** **`CSRMatrix`), too. One challenge is that the allocation size of sparse matrices depends on the number of non-zeros, which needs to be taken into account when identifying possible reuses.
* Make the DAPHNE compiler reorder the operations in the IR to make it more amenable for update-in-place operations. As an example, consider shift-and-scale, a widely used data preprocessing step in ML applications:** **`X = (X – mean(X, 1)) / stddev(X, 1);`. This subtracts the column mean from each value in a feature matrix** **`X` and divides all resulting values by the column standard deviation. This could result in the following (simplified) DaphneIR:
  ```mlir
  ...
  %5 = ... -> !daphne.Matrix<1000x100xf64>
  %6 = "daphne.meanCol"(%5) : (!daphne.Matrix<1000x100xf64>) -> !daphne.Matrix<1x100xf64>
  %7 = "daphne.ewSub"(%5, %6) : (!daphne.Matrix<1000x100xf64>, !daphne.Matrix<1x100xf64>) -> !daphne.Matrix<1000x100xf64>
  %8 = "daphne.stddevCol"(%5) : (!daphne.Matrix<1000x100xf64>) -> !daphne.Matrix<1x100xf64>
  %9 = "daphne.ewDiv"(%7, %8) : (!daphne.Matrix<1000x100xf64>, !daphne.Matrix<1x100xf64>) -> !daphne.Matrix<1000x100xf64>
  ...
  ```
  Here, the SSA value** **`%5` is the initial matrix** **`X` and** **`%9` is the resulting matrix** **`X` after the assignment. How many opportunities for update-in-place operation does this IR snippet offer? How could the operations be reordered to increase the number of opportunities? Try to generalize your observations and make them widely applicable for other examples.
