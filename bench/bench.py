#create a script that will run an application n times and aggregate the results in a boxplot
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

def run_command(update_in_place=False, args="../test.daph"):
    command = "../bin/daphne " + ("--update-in-place" if update_in_place else "") + " --timing " + args

    process = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stderr.splitlines()[-1]

def run_benchmarks(update_in_place=False, args="../test.daph", n=10):
    times = []
    for i in range(n):
        print("Running benchmark " + str(i) + "...")
        times.append(run_command(update_in_place, args))
    return times

def plot_boxplot(data, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.boxplot(data)
    plt.savefig(filename)

