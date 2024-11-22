from os import listdir
from os.path import isfile, join
import json
import pandas as pd
import matplotlib.pyplot as plt
import plot_config as pc
import numpy as np

d = "results/mesh_chain/"

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
        

x = list(timings_gr1.keys())
x.sort()
x = np.array(x, dtype="int")
y1 = [timings_gr1[ops]["tool"] for ops in x]
y2 = [timings_gr2[ops]["tool"] for ops in x]
y3 = [timings_gr3[ops]["tool"] for ops in x]

fig, ax = plt.subplots()

plt.rcParams['font.size'] = pc.font_size
plt.grid()

ax.plot(x, y1, marker='o', label="CTB #1", color=pc.edgecolors[2])
ax.plot(x, y2, marker='o', label="CTB #2", color=pc.edgecolors[3])
ax.plot(x, y3, marker='o', label="CTB #3", color=pc.edgecolors[4])

plt.legend()
ax.set_ylim([0, 0.82])

plt.xticks()

plt.xlabel("Number of operators")
plt.ylabel(pc.units["comp_time"]["label"])

plt.tight_layout(pad=0)
plt.savefig(d + f"lines.svg", format='svg') 


    
