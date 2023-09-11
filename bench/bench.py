import time
import sys
import subprocess
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import re
from matplotlib import colors as mcolors
from matplotlib import cm
import psutil

### GLOBALS ###
# Path to the application to run
DAPHNE_PATH = "../bin/daphne"
RESULT_PATH = "./results/"

def run_command(command, poll_interval=0.001): 

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   
    process_memory = psutil.Process(process.pid)
    peak_mem = 0
    while process.poll() is None:  # Check if the process has finished
        mem_info = process_memory.memory_info().rss // 1024  # Memory usage in KB
        if mem_info > peak_mem:
            peak_mem = mem_info

        time.sleep(poll_interval)

    #print(f"Peak memory usage: {peak_mem} KB")

    _, stderr = process.communicate()

    return peak_mem, stderr.decode().splitlines()[-1]

def run_benchmark(update_in_place, args, file_path, n):

    if update_in_place:
        command = [DAPHNE_PATH] + args + ["--update-in-place", "--timing"] + [file_path]
    else:
        command = [DAPHNE_PATH] + args + ["--timing"] + [file_path]

    print("Running benchmark " + str(n) + " time(s)")
    print("with command: " + " ".join(command))

    arr_timing = []
    arr_peak_mem = []
    for i in range(n):
        print("Running benchmark " + str(i) + "...")
        peak_mem, timing = run_command(command)
        arr_timing.append(json.loads(timing))
        arr_peak_mem.append(peak_mem)

    return {
        "command": command,
        "n": n,
        "update_in_place": update_in_place,
        "peak_mem": arr_peak_mem,
        "timing": arr_timing
    }

def save_dict_to_json(dict, file_path):
    with open(RESULT_PATH + file_path, 'w') as fp:
        json.dump(dict, fp)

def create_file_name(file_path, prefix, n, update_in_place):
    file_name = file_path.split("/")[-1].split(".")[0]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime((time.time())))

    #creaate dir if not exists
    if not os.path.exists(RESULT_PATH + file_name):
       os.makedirs(RESULT_PATH + file_name) 

    return file_name + "/" + prefix + "_" + str(n) + "_" + str(update_in_place) + "_" + timestamp + ".json"

def benchit(file_path, prefix, n, update_in_place, args):
    result = run_benchmark(update_in_place, args, file_path, n)
    file_name = create_file_name(file_path, prefix, n, update_in_place)
    save_dict_to_json(result, file_name)

    return result

### NORMALIZE MATRIX ###

#benchit(file_path="./normalize_matrix.daph", prefix="small", n=100, update_in_place=True, args=["--args", "n=50"])
#benchit(file_path="./normalize_matrix.daph", prefix="small", n=100, update_in_place=False, args=["--args", "n=50"])

#benchit(file_path="./normalize_matrix.daph", prefix="medium", n=25, update_in_place=True, args=["--args", "n=5000"])
#benchit(file_path="./normalize_matrix.daph", prefix="medium", n=25, update_in_place=False, args=["--args", "n=5000"])

#benchit(file_path="./normalize_matrix.daph", prefix="large", n=2, update_in_place=True, args=["--args", "n=25000"])
#benchit(file_path="./normalize_matrix.daph", prefix="large", n=2, update_in_place=False, args=["--args", "n=25000"])

#benchit(file_path="./normalize_matrix.daph", prefix="out-of-memory", n=1, update_in_place=True, args=["--args", "n=36500"])
#benchit(file_path="./normalize_matrix.daph", prefix="out-of-memory", n=1, update_in_place=False, args=["--args", "n=36500"]) # <-- will crash

### TRANSPOSE ADDITION ###

#benchit(file_path="./transpose_addition.daph", prefix="small", n=100, update_in_place=True, args=["--args", "n=50"])
#benchit(file_path="./transpose_addition.daph", prefix="small", n=100, update_in_place=False, args=["--args", "n=50"])

#benchit(file_path="./transpose_addition.daph", prefix="medium", n=25, update_in_place=True, args=["--args", "n=5000"])
#benchit(file_path="./transpose_addition.daph", prefix="medium", n=25, update_in_place=False, args=["--args", "n=5000"])

#benchit(file_path="./transpose_addition.daph", prefix="large", n=2, update_in_place=True, args=["--args", "n=25000"])
#benchit(file_path="./transpose_addition.daph", prefix="large", n=2, update_in_place=False, args=["--args", "n=25000"])

### ADDITION ###

#benchit(file_path="./addition.daph", prefix="small", n=100, update_in_place=True, args=["--args", "n=50"])
#benchit(file_path="./addition.daph", prefix="small", n=100, update_in_place=False, args=["--args", "n=50"])

#benchit(file_path="./addition.daph", prefix="medium", n=25, update_in_place=True, args=["--args", "n=5000"])
#benchit(file_path="./addition.daph", prefix="medium", n=25, update_in_place=False, args=["--args", "n=5000"])

#benchit(file_path="./addition.daph", prefix="large", n=2, update_in_place=True, args=["--args", "n=25000"])
#benchit(file_path="./addition.daph", prefix="large", n=2, update_in_place=False, args=["--args", "n=25000"])


benchit(file_path="./addition_readMatrix.daph", prefix="small", n=100, update_in_place=True, args=["--args", "n=\"X_small.csv\""])
benchit(file_path="./addition_readMatrix.daph", prefix="small", n=100, update_in_place=False, args=["--args", "n=\"X_small.csv\""])

benchit(file_path="./addition_readMatrix.daph", prefix="medium", n=25, update_in_place=True, args=["--args", "n=\"X_medium.csv\""])
benchit(file_path="./addition_readMatrix.daph", prefix="medium", n=25, update_in_place=False, args=["--args", "n=\"X_medium.csv\""])

benchit(file_path="./addition_readMatrix.daph", prefix="large", n=2, update_in_place=True, args=["--args", "n=\"X_large.csv\""])
benchit(file_path="./addition_readMatrix.daph", prefix="large", n=2, update_in_place=False, args=["--args", "n=\"X_large.csv\""])