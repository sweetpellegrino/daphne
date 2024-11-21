import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plot_config as pc

exp_folder = "results/ae-malloc/"

key = "tool.malloc"
unit = 1e9
unit_text = "(Mean) Allocated Data [GB]"
filename = "ae-malloc.png"

with open(exp_folder + "timings.json", "r") as f:
    data = json.load(f)

fig, axs = plt.subplots(1, 1, figsize=(pc.figsize_width, pc.figsize_height))
plt.rcParams['font.size'] = pc.font_size

def draw_bars(ax, x, y, y4):

    x2 = x + pc.bar_width
    x2[0] = x[0] 
    y4[0] = y[0]

    handle = ax.bar(x, y, pc.bar_width, color=pc.colors, edgecolor=pc.edgecolors)
    handle2 = ax.bar(x2, y4, pc.bar_width, color=pc.edgecolors, edgecolor=pc.edgecolors)

    if False:
        for i,h in enumerate(handle + handle2):
            if i == len(handle):
                continue
            ax.text(h.get_x() + h.get_width() / 2.0, h.get_height() + 0.15, f"{h.get_height():.2f}", color='black', ha='center')

def calc_means(d_exec):
    y = []
    names = []
    for j, c in enumerate(d_exec):
            
        timings = pd.DataFrame(c["timings"])
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

    x[0] = x[0] - pc.bar_width/2 
    plt.xticks(x+pc.bar_width/2, pc.xticks_name)

    #ax.set_title(script_args)
    ax.set_ylabel(unit_text)

plt.tight_layout()

plt.savefig(exp_folder + filename + ".png")
plt.savefig(exp_folder + filename + ".svg", format='svg')