from os import listdir
from os.path import isfile, join
import json
import pandas as pd
import matplotlib.pyplot as plt
import plot_config as pc
import numpy as np

exp_name = "agg"
d = f"results/horz/{exp_name}/48_extreme/"
d4 = f"results/horz/{exp_name}/48_extreme/"

files = [f for f in listdir(d) if isfile(join(d, f))]

timings_hf = {}
timings_no_hf = {}
for file in files:
    if file[-3:] == "png" or file[-3:] == "svg":
        continue

    with open(d + file, "r") as f:
        data = json.load(f)
        num_ops = data["settings"]["num-ops"]
        _timings = data["execs"][0]["timings"]
        
        _t_df = pd.DataFrame(_timings)
        mean_time = _t_df["tool"].mean() / pc.units["exec_time"]["conversion"]
        #peak_mem = _t_df["peak_rss_kilobytes"].max() / 1e3

        timings_hf[num_ops] = {
            "tool" : mean_time,
         #   "peak_rss_kilobytes" : peak_mem
        }

        _timings = data["execs"][1]["timings"]
        
        _t_df = pd.DataFrame(_timings)
        mean_time = _t_df["tool"].mean() / pc.units["exec_time"]["conversion"]
        #peak_mem = _t_df["peak_rss_kilobytes"].max() / 1e3

        timings_no_hf[num_ops] = {
            "tool" : mean_time,
        #    "peak_rss_kilobytes" : peak_mem
        }

files = [f for f in listdir(d4) if isfile(join(d4, f))]

timings_hf_4 = {}
timings_no_hf_4 = {}
for file in files:
    if file[-3:] == "png" or file[-3:] == "svg":
        continue

    with open(d4 + file, "r") as f:
        data = json.load(f)
        num_ops = data["settings"]["num-ops"]
        _timings = data["execs"][0]["timings"]
        
        _t_df = pd.DataFrame(_timings)
        mean_time = _t_df["tool"].mean() / pc.units["exec_time"]["conversion"]
        #peak_mem = _t_df["peak_rss_kilobytes"].max() / 1e3

        timings_hf_4[num_ops] = {
            "tool" : mean_time,
         #   "peak_rss_kilobytes" : peak_mem
        }

        _timings = data["execs"][1]["timings"]
        
        _t_df = pd.DataFrame(_timings)
        mean_time = _t_df["tool"].mean() / pc.units["exec_time"]["conversion"]
        #peak_mem = _t_df["peak_rss_kilobytes"].max() / 1e3

        timings_no_hf_4[num_ops] = {
            "tool" : mean_time,
        #    "peak_rss_kilobytes" : peak_mem
        }

x = list(timings_hf.keys())
x.sort()
x = np.array(x, dtype="int")
y1 = [timings_hf[ops]["tool"] / ops for ops in x] 
y2 = [timings_no_hf[ops]["tool"] / ops for ops in x]
y3 = [timings_hf_4[ops]["tool"] / ops for ops in x]
y4 = [timings_no_hf_4[ops]["tool"] / ops for ops in x]

fig, ax = plt.subplots()
plt.rcParams['font.size'] = 12

ax.plot(x, y3, marker='*', label="48 threads with horz. fusion", color=pc.edgecolors[7])
ax.plot(x, y4, marker='^', label="48 threads w/o horz. fusion", color=pc.edgecolors[8])

plt.grid()
plt.legend()

plt.xticks(np.arange(x.min(), x.max() + 1, 1))

plt.xlabel("Number of operators")
plt.ylabel("(Mean) Execution Time per operator [s]")

plt.tight_layout(pad=0)
plt.savefig(d + f"{exp_name}-lines.svg", format='svg') 


    
