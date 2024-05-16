import os
import subprocess
import time
import numpy as np

DAPHNE_PATH = "../../daphne-X86-64-1-bin/bin/daphne"

PAPI_EVENTS = "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE-MISSES,perf::BRANCHES,perf::BRANCH-MISSES"
PAPI_REPORT = 1

NUM_RUNS=3
STEPS=100

WIDTH=100
HEIGHT=100000

#https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
def gen_log_space(limit, n):
    n = n+1
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)[1:]

name = str(HEIGHT) + "_" + str(WIDTH)
os.makedirs(name, exist_ok=True)
os.chdir(name)

logspace = gen_log_space(HEIGHT, STEPS)

for batchSize in logspace:

    args = [DAPHNE_PATH, "--num-threads=1", "--vec", f"--batchSize={batchSize}", "--args", f"n={HEIGHT},m={WIDTH}", f"../../home-niklas/daphne/sketch/batchsize/n_m_matrix_unary_sqrt.daph"]

    for _ in range(NUM_RUNS):

        os.makedirs(str(batchSize), exist_ok=True)
        os.chdir(str(batchSize))

        print("\033[91m" + f"Running {HEIGHT} {WIDTH} {batchSize}" + "\033[0m")

        os.environ["PAPI_EVENTS"] = PAPI_EVENTS
        os.environ["PAPI_REPORT"] = str(PAPI_REPORT)

        subprocess.run(args, check=True)
        time.sleep(1)

os.chdir("..")