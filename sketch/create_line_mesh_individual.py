from os import listdir
from os.path import isfile, join
import json
import pandas as pd
import matplotlib.pyplot as plt
import plot_config as pc
import numpy as np

exp_name = "mesh_mesh"
d = f"results/scaling/{exp_name}/"

files = [f for f in listdir(d) if isfile(join(d, f))]

def extract_means(data):
    _timings = data["timings"]
    _t_df = pd.DataFrame(_timings)
    mean_time = _t_df["compilation_seconds"].mean() / pc.units["comp_time"]["conversion"]
    std = _t_df["compilation_seconds"].std() / pc.units["comp_time"]["conversion"]

    return { "tool": mean_time, "std": std }

names = ["CTB #1", "CTB #2", "CTB #3", "Base", "BVec"]
colors = [2, 3, 4, 0, 1]
markers = ["+", "^", "s", "x", "o"]
for i in range(0,5):
    timings = {}

    if i <= 2:
        for file in files:
            if file[-3:] == "png" or file[-3:] == "svg":
                continue

            with open(d + file, "r") as f:
                data = json.load(f)
                num_ops = data["settings"]["depth"] * data["settings"]["width"]
                timings[num_ops] = extract_means(data["execs"][i])
    elif i == 3:
        d_base = d + "base/"
        files = [f for f in listdir(d_base) if isfile(join(d_base, f))]
        for file in files:
            if file[-3:] == "png" or file[-3:] == "svg":
                continue

            with open(d_base + file, "r") as f:
                data = json.load(f)
                num_ops = data["settings"]["depth"] * data["settings"]["width"]
                timings[num_ops] = extract_means(data["execs"][0]) 
    elif i == 4:
        print(i)
        d_base = d + "bvec/"
        files = [f for f in listdir(d_base) if isfile(join(d_base, f))]
        for file in files:
            if file[-3:] == "png" or file[-3:] == "svg":
                continue

            with open(d_base + file, "r") as f:
                data = json.load(f)
                num_ops = data["settings"]["depth"] * data["settings"]["width"]
                timings[num_ops] = extract_means(data["execs"][0])
            
    x = list(timings.keys())
    x.sort()
    x = np.array(x, dtype="int")
    y1 = [timings[ops]["tool"] for ops in x]
    y1 = np.array(y1, dtype="float32")
    std = [timings[ops]["std"] for ops in x]
    std = np.array(std, dtype="float32")

    fig, ax = plt.subplots()

    plt.rcParams['font.size'] = pc.font_size
    plt.grid()

    ax.plot(x, y1, marker=markers[i], label=names[i], color=pc.edgecolors[colors[i]])
    y_u = y1 + std
    y_l = y1 - std
    plt.fill_between(x, y_l, y_u, color=pc.colors[colors[i]], alpha=0.5, label='Standard Deviation')

    #ax.set_ylim([0, 7.2])
    ax.set_ylim([0, 60])

    plt.legend()
    plt.xticks()

    plt.xlabel("Number of operators")
    plt.ylabel(pc.units["comp_time"]["label"])

    plt.tight_layout(pad=0)
    plt.savefig(d + f"{exp_name}-gr{i+1}-line.svg", format='svg') 


    
