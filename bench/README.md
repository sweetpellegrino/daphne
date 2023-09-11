# Benchmarking

This document provides key information about benchmarking the implemented in-place update approach to reproduce our results.

**E2E-Benchmark:**

There are currently following cases with different matrix sizes benchmarked:

* [addition.daph](addition.daph)
* [addition_readMatrix.daph](addition_readMatrix.daph)
* [normalize_matrix.daph](normalize_matrix.daph)
* [tranpose.daph](tranpose.daph)
* [kmeans.daph](kmeans.daph)

 We are capturing the following indicators:

* *timings* from DAPHNE by using `--timing` as an argument.
* *Peak memory* *consumption*

To execute the benchmark, run `﻿$ python3 bench.py.` Prior to that, it is necessary to run [create_matrix_files.daph](create_matrix_files.daph) in order to generate static matrix files that will be stored on disk. The resulting matrices have a total size of 2.3GB and are used in *addition_readMatrix.daph*.

The fine-granular results can be displayed as boxplots with the [draw_graphs.ipynb](draw_graphs.ipynb).

**Microbenchmark:**

The implemented kernels are tested directly in Catch2 using the BENCHMARK feature. This demonstrates the impact of not allocating new memory for a data object, especially in combination when another algorithm is used.

The tag `[inplace-bench]` is deactivated by default, we can run it by executing:

```bash
$ ./test.sh [inplace-bench]
```

The result of the run on the bench VM can be found in file: XXXX

## System Information

The benchmark was done inside a VM on hetzner.cloud

Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
Address sizes:                      40 bits physical, 48 bits virtual
CPU(s):                             4
On-line CPU(s) list:                0-3
Thread(s) per core:                 1
Core(s) per socket:                 4
Socket(s):                          1
NUMA node(s):                       1
Vendor ID:                          GenuineIntel
CPU family:                         6
Model:                              85
Model name:                         Intel Xeon Processor (Skylake, IBRS)
Stepping:                           4
CPU MHz:                            2099.998
BogoMIPS:                           4199.99
Hypervisor vendor:                  KVM
Virtualization type:                full
L1d cache:                          128 KiB
L1i cache:                          128 KiB
L2 cache:                           16 MiB
L3 cache:                           16 MiB

RAM: 16GB
Disk Storage: SSD 160GB
