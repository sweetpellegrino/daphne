import os
import subprocess
import concurrent.futures
import time

j=1
numbers = [0, 108, 156, 180, 204, 216, 240, 252, 264, 270, 294, 304, 318, 48, 72, 96]
script_path = "./sketch/n_matrix_multi_complex_connected_leafs_large.daph"
time_suffix = int(time.time())

def run_command(script_path, i):
    command = ["./bin/daphne", "--timing", "--vec", "--vec-type", "ONE", "--run-key", str(i*j), "--num-threads=1", script_path]
    print(" ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return i, stdout.decode(), stderr.decode()

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for i in numbers:
            futures.append(executor.submit(run_command, script_path, i))
        for future in concurrent.futures.as_completed(futures):
            i, stdout, stderr = future.result()
            print(script_path, i, stdout, stderr)
   