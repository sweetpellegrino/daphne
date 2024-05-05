#!/bin/bash

NUM_RUNS=2

PAPI_EVENTS="perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE-MISSES,perf::BRANCHES,perf::BRANCH-MISSES"
PAPI_REPORT=1

PROGRAM_PATH="../../../daphne-X86-64-1-bin/bin/daphne"

# Define an array to store experiment settings
experiments=("sym" "long" "wide")
subexperiments=("csr_aggCol" "csc_aggCol" "csr_aggRow" "csc_aggRow")
experiments_args_sym=("--select-matrix-repr  --force-sparse=csrcsr  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggCol_sym.daph" 
		        "--select-matrix-repr  --force-sparse=csccsc  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggCol_sym.daph" 
		        "--select-matrix-repr  --force-sparse=csrcsr  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggRow_sym.daph" 
		        "--select-matrix-repr  --force-sparse=csccsc  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggRow_sym.daph")

experiments_args_long=("--select-matrix-repr  --force-sparse=csrcsr  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggCol_long.daph" 
		  "--select-matrix-repr  --force-sparse=csccsc  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggCol_long.daph" 
		  "--select-matrix-repr  --force-sparse=csrcsr  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggRow_long.daph" 
		  "--select-matrix-repr  --force-sparse=csccsc  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggRow_long.daph")

experiments_args_wide=("--select-matrix-repr  --force-sparse=csrcsr  --explain=select_matrix_repr  --args dim=1000 --timing ../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggCol_wide.daph" 
		  "--select-matrix-repr  --force-sparse=csccsc  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggCol_wide.daph" 
		  "--select-matrix-repr  --force-sparse=csrcsr  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggRow_wide.daph" 
		  "--select-matrix-repr  --force-sparse=csccsc  --explain=select_matrix_repr  --args dim=1000 --timing ../../../home-niklas/daphne/sketch/csc_csr/n_matrix_aggRow_wide.daph")

# Loop through each experiment
for i in "${!experiments[@]}"; do

    dir_name=${experiments[i]}
    mkdir -p "$dir_name"
    cd ${experiments[i]}

    experiments_arg_list=()
    case ${experiments[i]} in
        "sym") experiments_arg_list=$experiments_args_sym ;;
        "long") experiments_arg_list=$experiments_args_long ;;
        "wide") experiments_arg_list=$experiments_args_wide ;;
    esac

    for k in "${!subexperiments[@]}"; do

        dir_name=${subexperiments[k]}
        mkdir -p "$dir_name"
        cd ${subexperiments[k]}

        for j in $(seq 1 $NUM_RUNS); do

            export PAPI_EVENTS
            export PAPI_REPORT

            # Run the program with its arguments
            echo ${experiments[i]} ${subexperiments[k]}
            echo ${experiments_arg_list[k]}
            $PROGRAM_PATH ${experiments_arg_list[k]}

            sleep 1
        done
        cd ..
    done
    cd ..
done
