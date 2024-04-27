import time
import sys
import subprocess
import os
import time
import json
import psutil
import re

### GLOBALS ###
# Path to the application to run
DAPHNE_PATH = "../bin/daphne"
#DAPHNE_PATH = "../bin/daphne-96fbecb"
RESULT_PATH = "./results/"
#RESULT_PATH = "./results-96fbecb/"

### ARGUMENTS ###
no_overwrite = False
if "--no-overwrite" in sys.argv:
    no_overwrite = True
use_perf = False
if "--use-perf" in sys.argv:
    use_perf = True

### FUNCTIONS ###
def run_command(command, poll_interval=0.001): 

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   
    process_memory = psutil.Process(process.pid)
    peak_mem = 0
    while process.poll() is None: 
        mem_info = process_memory.memory_info().rss // 1024  # Memory usage in KB
        if mem_info > peak_mem:
            peak_mem = mem_info

        time.sleep(poll_interval)

    #print(f"Peak memory usage: {peak_mem} KB")

    stdout, stderr = process.communicate()
    return peak_mem, stdout.decode(), stderr.decode()


def extract_perf_stats(input_string):

    return {
        "cycles":               re.search(r'(\d+)\W+cycles', input_string).group(1),
        "instructions":         re.search(r'(\d+)\W+instructions', input_string).group(1),
        "cache-references":     re.search(r'(\d+)\W+cache-references', input_string).group(1),
        "cache-misses":         re.search(r'(\d+)\W+cache-misses', input_string).group(1),
        "seconds time elapsed": re.search(r'(\d+\.\d+)\W+seconds time elapsed', input_string).group(1),
        "seconds user":         re.search(r'(\d+\.\d+)\W+seconds user', input_string).group(1),
        "seconds sys":          re.search(r'(\d+\.\d+)\W+seconds sys', input_string).group(1),
    }

def run_benchmark(args, file_path, n, use_perf):

    command = []

    if use_perf:
        command += ["perf", "stat", "-e", "cycles,instructions,cache-references,cache-misses"]

    command += [DAPHNE_PATH] + args + ["--timing"] + [file_path]

    print("Running benchmark " + str(n) + " time(s)")
    print("with command: " + " ".join(command))

    arr_timing = []
    arr_peak_mem = []
    arr_perf_stats = []
    for i in range(n):
        print("Running benchmark " + str(i) + "...", end='\r')
        peak_mem, stdout, stderr = run_command(command)

        if use_perf:
            arr_perf_stats.append(extract_perf_stats(stderr))
            arr_timing.append(json.loads(stderr.splitlines()[0]))
            arr_peak_mem.append(peak_mem)
        else:
            arr_timing.append(json.loads(stderr.splitlines()[-1]))
            arr_peak_mem.append(peak_mem)

    return {
        "command": command,
        "n": n,
        "peak_mem": arr_peak_mem,
        "perf_stats": arr_perf_stats,
        "timing": arr_timing
    }

def save_dict_to_json(dict, file_path, prefix, n, use_perf, suffix):

    new_file = create_file_name(file_path, prefix, n, use_perf, suffix)
    result_dir_script = file_path.split("/")[-1].split(".")[0]

    if not os.path.exists(RESULT_PATH + result_dir_script):
       os.makedirs(RESULT_PATH + result_dir_script) 

    if os.path.exists(RESULT_PATH + new_file):
        file_stats = os.stat(RESULT_PATH + new_file)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime((file_stats.st_ctime)))
        old_file = create_file_name(file_path, prefix, n, timestamp, suffix)
        
        if not os.path.exists(RESULT_PATH + "archive/" + result_dir_script):
            os.makedirs(RESULT_PATH + "archive/" + result_dir_script) 
        os.rename(RESULT_PATH + new_file, RESULT_PATH + "archive/" + old_file)

    with open(RESULT_PATH + new_file, 'w') as fp:
        json.dump(dict, fp)

def create_file_name(file_path, prefix, n, use_perf, suffix):
    file_name = file_path.split("/")[-1].split(".")[0]
    if suffix:
        return file_name + "/" + prefix + "_" + str(n) + "_" + str(use_perf) + "_" + suffix + ".json"
    else:
        return file_name + "/" + prefix + "_" + str(n) + "_" + str(use_perf) + ".json"

# n: number of samples
def benchit(file_path, prefix, n, use_perf, args, suffix=""):
    print("### BENCHMARKING " + file_path + " ###")
    if (no_overwrite and os.path.exists(RESULT_PATH + create_file_name(file_path, prefix, n, use_perf, suffix))):
        print("Benchmark already exists. Skipping...")
        return
    result = run_benchmark(args, file_path, n, use_perf)
    save_dict_to_json(result, file_path, prefix, n, use_perf, suffix)
    print("Done!")

    return result

### BENCHMARKS ###

for i in range(0, 15):
    dim = pow(2, i)
    benchit(file_path="./n_matrix.daph", prefix=f"2_{i}", n=10, use_perf=use_perf, args=["--args", f"dim={dim}"])