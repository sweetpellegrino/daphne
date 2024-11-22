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

for i in range(0,3):

    def extract_means(data):
        _timings = data["timings"]
        _t_df = pd.DataFrame(_timings)
        mean_time = _t_df["compilation_seconds"].mean() / pc.units["comp_time"]["conversion"]
        std = _t_df["compilation_seconds"].std() / pc.units["comp_time"]["conversion"]

        return { "tool": mean_time, "std": std }

    for file in files:
        if file[-3:] == "png" or file[-3:] == "svg":
            continue

        with open(d + file, "r") as f:
            data = json.load(f)
            num_ops = data["settings"]["depth"] * data["settings"]["width"]

            timings_gr1[num_ops] = extract_means(data["execs"][i])
            

    x = list(timings_gr1.keys())
    x.sort()
    x = np.array(x, dtype="int")
    y1 = [timings_gr1[ops]["tool"] for ops in x]
    y1 = np.array(y1, dtype="float32")
    std = [timings_gr1[ops]["std"] for ops in x]
    std = np.array(std, dtype="float32")

    fig, ax = plt.subplots()

    plt.rcParams['font.size'] = pc.font_size
    plt.grid()

    ax.plot(x, y1, marker='o', label=f"CTB #{i+1}", color=pc.edgecolors[2+i])
    y_u = y1 + std
    y_l = y1 - std
    plt.fill_between(x, y_l, y_u, color=pc.colors[2+i], alpha=0.5, label='Standard Deviation')
    ax.set_ylim([0, 0.82])

    plt.legend()
    plt.xticks()

    plt.xlabel("Number of operators")
    plt.ylabel(pc.units["comp_time"]["label"])

    plt.tight_layout(pad=0)
    plt.savefig(d + f"gr{i+1}-line.svg", format='svg') 


    
