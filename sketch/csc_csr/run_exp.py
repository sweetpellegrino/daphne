import os
import subprocess
import time

DAPHNE_PATH = "../../../../daphne-X86-64-1-bin/bin/daphne"

PAPI_EVENTS = "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE-MISSES,perf::BRANCHES,perf::BRANCH-MISSES"
PAPI_REPORT = 1

NUM_RUNS=10
LEN_DIM = [100, 750, 1000, 10000] 

experiments = ["sym", "long", "wide"]
variants_sparse = ["csrcsr", "csccsc"]
variants_direction = ["aggCol", "aggRow"]

for len_dim in LEN_DIM:
    os.makedirs(str(len_dim), exist_ok=True)
    os.chdir(str(len_dim))

    for exp in experiments:
        os.makedirs(exp+"-"+str(len_dim), exist_ok=True)
        os.chdir(exp+"-"+str(len_dim))

        for sp_repr in variants_sparse:

            for agg_dir in variants_direction:
                os.makedirs(sp_repr+"-"+agg_dir, exist_ok=True)
                os.chdir(sp_repr+"-"+agg_dir)
                
                args = [DAPHNE_PATH, "--select-matrix-repr", f"--force-sparse={sp_repr}", "--args", f"dim={len_dim}", "--timing", f"../../../../home-niklas/daphne/sketch/csc_csr/n_matrix_{agg_dir}_{exp}.daph"]

                for _ in range(NUM_RUNS): 
                    print("\033[91m" + f"Running {exp} {sp_repr} {agg_dir}" + "\033[0m")

                    os.environ["PAPI_EVENTS"] = PAPI_EVENTS
                    os.environ["PAPI_REPORT"] = str(PAPI_REPORT)

                    subprocess.run(args, check=True)
                    time.sleep(1)

                os.chdir("..")
        os.chdir("..")
    os.chdir("..")