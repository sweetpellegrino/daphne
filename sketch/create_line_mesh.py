from os import listdir
from os.path import isfile, join
import json
import pandas as pd
import matplotlib.pyplot as plt
import plot_config as pc
import numpy as np

exp_name = "mesh_chain"
d = f"results/scaling/{exp_name}/"

files = [f for f in listdir(d) if isfile(join(d, f))]

timings_gr1 = {}
timings_gr2 = {}
timings_gr3 = {}

def extract_means(data):
    _timings = data["timings"]
    _t_df = pd.DataFrame(_timings)
    mean_time = _t_df["compilation_seconds"].mean() / pc.units["comp_time"]["conversion"]

    return { "tool": mean_time }

for file in files:
    if file[-3:] == "png" or file[-3:] == "svg":
        continue

    with open(d + file, "r") as f:
        data = json.load(f)
        num_ops = data["settings"]["depth"] * data["settings"]["width"]

        timings_gr1[num_ops] = extract_means(data["execs"][0])
        timings_gr2[num_ops] = extract_means(data["execs"][1])
        timings_gr3[num_ops] = extract_means(data["execs"][2])

d_base = d + "base/"
files = [f for f in listdir(d_base) if isfile(join(d_base, f))]
timings_base = {}
for file in files:
    if file[-3:] == "png" or file[-3:] == "svg":
        continue

    with open(d_base + file, "r") as f:
        data = json.load(f)
        num_ops = data["settings"]["depth"] * data["settings"]["width"]
        timings_base[num_ops] = extract_means(data["execs"][0])

d_base = d + "bvec/"
files = [f for f in listdir(d_base) if isfile(join(d_base, f))]
timings_bvec = {}
for file in files:
    if file[-3:] == "png" or file[-3:] == "svg":
        continue

    with open(d_base + file, "r") as f:
        data = json.load(f)
        num_ops = data["settings"]["depth"] * data["settings"]["width"]
        timings_bvec[num_ops] = extract_means(data["execs"][0])

x = list(timings_gr1.keys())
x.sort()
x = np.array(x, dtype="int")
y1 = [timings_gr1[ops]["tool"] for ops in x]
y2 = [timings_gr2[ops]["tool"] for ops in x]
y3 = [timings_gr3[ops]["tool"] for ops in x]
y4 = [timings_base[ops]["tool"] for ops in x]
y5 = [timings_bvec[ops]["tool"] for ops in x]

fig, ax = plt.subplots()

plt.rcParams['font.size'] = pc.font_size
plt.grid()

ax.plot(x, y4, marker='x', label="Base", color=pc.edgecolors[0])
ax.plot(x, y5, marker='o', label="BVec", color=pc.edgecolors[1])
ax.plot(x, y1, marker='+', label="CTB #1", color=pc.edgecolors[2])
ax.plot(x, y2, marker='^', label="CTB #2", color=pc.edgecolors[3])
ax.plot(x, y3, marker='s', label="CTB #3", color=pc.edgecolors[4])

plt.legend()
ax.set_ylim([0, 7.2])
#ax.set_ylim([0, 60])

plt.xticks()

plt.xlabel("Number of operators")
plt.ylabel(pc.units["comp_time"]["label"])

plt.tight_layout(pad=0)
plt.savefig(d + f"{exp_name}-lines.svg", format='svg') 


    
