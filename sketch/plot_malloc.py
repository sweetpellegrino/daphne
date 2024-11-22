import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plot_config as pc
import matplotlib.patches as mpatches

exp_name = "ae"
exp_folder = f"results/microbenchmark/{exp_name}/{exp_name}-malloc/"

key = "tool.malloc"
unit = 1e9
unit_text = "(Mean) Allocated Data [GB]"
filename = f"{exp_name}-malloc"

with open(exp_folder + "timings.json", "r") as f:
    data = json.load(f)

fig, axs = plt.subplots(1, 1, figsize=(pc.figsize_width, pc.figsize_height))
plt.rcParams['font.size'] = pc.font_size
 
legend_handles = [
    mpatches.Patch(facecolor=pc.colors[0], edgecolor=pc.edgecolors[0], label='Scalar'),
    mpatches.Patch(facecolor=pc.colors[0], edgecolor=pc.edgecolors[0], hatch='///', label='1 Thread'),
]

plt.legend(loc='center right', handles=legend_handles)

def draw_bars(ax, x, y):

    handle = ax.bar(x, y, pc.bar_width, color=pc.colors, edgecolor=pc.edgecolors)

    for i,h in enumerate(handle):
        if i != 0:
            h.set_hatch("/")

    if False:
        for i,h in enumerate(handle + handle2):
            if i == len(handle):
                continue
            ax.text(h.get_x() + h.get_width() / 2.0, h.get_height() + 0.15, f"{h.get_height():.2f}", color='black', ha='center')

def calc_means(d_exec):
    y = []
    names = []
    for j, c in enumerate(d_exec):
            
        timings = pd.json_normalize(c["timings"], sep=".")
        
        mean = timings[key].mean() / unit
        std = timings[key].std() / unit

        name = "".join([s[-1] for s in c["cmd"]])
        names.append(name)
        y.append(mean)

    return y, names

for i, d in enumerate(data):
    script_args = " ".join([d["script"]["path"]] + d["script"]["args"])

    if len(data) == 1:
        ax = axs
    else:
        ax = axs[i]
    
    x = np.arange(0.0, len(d["exec"]))
    x[0] = x[0] + pc.bar_width/2 

    y, names = calc_means(d["exec"])
    
    draw_bars(ax, x, y)

    _max = np.max(y)
    
    ax.set_ylim(0, _max + pc.offset_max*_max)

    plt.xticks(x, pc.xticks_name)

    #ax.set_title(script_args)
    ax.set_ylabel(unit_text)

plt.tight_layout(pad=0)

plt.savefig(exp_folder + filename + ".svg", format='svg')