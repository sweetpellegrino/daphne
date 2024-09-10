import subprocess
import concurrent.futures
import time

def run_command(i):
    #command = ["../bin/daphne","--timing", "--vec", "--vec-type", "ONE", "--run-key", str(i), "--num-threads=1", "../test/api/cli/algorithms/kmeans.daphne", "r=1", "f=1", "c=1", "i=1"]
    command = ["../bin/daphne","--timing", "--vec", "--vec-type", "ONE", "--run-key", str(i), "--num-threads=1", "n_matrix_multi_complex_connected_leafs_large.daph"]
    print(" ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode()

n = 72
exec_range = range(0, n)
check_interval = 5


#further improv: https://www.xmodulo.com/run-program-process-specific-cpu-cores-linux.html
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in exec_range:
        futures.append(executor.submit(run_command, i))
    for future in concurrent.futures.as_completed(futures):
        print(future.result())