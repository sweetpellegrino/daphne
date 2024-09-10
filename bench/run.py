import os
import subprocess
import concurrent.futures
import time

n = 10
j = 7
exec_range = range(0, n)
script_path = "../sketch/n_matrix_multi_complex_connected_leafs_large.daph"
time_suffix = int(time.time())

def run_command(script_path, i):
    command = ["../bin/daphne", "--timing", "--vec", "--vec-type", "ONE", "--run-key", str(i*j), "--num-threads=1", script_path]
    print(" ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return i, stdout.decode(), stderr.decode()

def save_results(script_path, i, stdout, stderr):
    base_name = os.path.basename(script_path) + "_" + str(time_suffix)
    result_dir = os.path.join('results', f'{base_name}/graph-{i*j}')
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "stdout.txt"), "w") as f:
        f.write(stdout)
    with open(os.path.join(result_dir, "stderr.txt"), "w") as f:
        f.write(stderr)

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for i in exec_range:
            futures.append(executor.submit(run_command, script_path, i))
        for future in concurrent.futures.as_completed(futures):
            i, stdout, stderr = future.result()
            save_results(script_path, i, stdout, stderr)
   