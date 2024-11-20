import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

exp_folder = "results/r-1-3/"
exp_folder_4 = "results/r-4-3/"

ignore_ixs = []
if False: 
    key = "tool"
    unit = 1e9
    unit_text = "(Mean) Execution Seconds"
else: 
    key = "peak_rss_kilobytes"
    unit = 1e3
    unit_text = "(Mean) Peak Resident Set Size"


with open(exp_folder + "timings.json", "r") as f:
    data = json.load(f)
with open(exp_folder_4 + "timings.json", "r") as f:
    data_4 = json.load(f)

nrows = len(data)

fig, axs = plt.subplots(nrows, 1, figsize=(8, 4*nrows))

# gray 
# (245,245,245) : #f5f5f5
# (102,102,102) : #666666
# blue
# (219,232,251) : #dbe8fb
# (110,143,189) : #6e8fbd
# green
# (214,232,213) : #d6e8d5
# (131,178,106) : #83b26a
# orange
# (255,230,206) : #ffe6ce
# (214,154,35) : #d69a23
# red
# (247,206,206) : #f7cece
# (182,85,82) : #b65552
# purple
# (225,214,231) :#e1d6e7
# (150,116,165) :#9674a5

#colors = ["tab:gray", "seagreen", "seagreen", "mediumseagreen", "mediumseagreen", "springgreen", "springgreen"]
colors = ["#f5f5f5", "#dbe8fb", "#d6e8d5"]
edgecolors = ["#666666", "#6e8fbd", "#83b26a"]

legend = ""

def draw_bars(ax, x, y, y4):
    width = 0.35
    handle = ax.bar(x, y, width, color=colors, edgecolor=edgecolors)
    handle2 = ax.bar(x + width, y4, width, color=edgecolors, edgecolor=edgecolors)

    for h in handle + handle2:
        height = h.get_height()
        ax.text(h.get_x() + h.get_width() / 2.0, h.get_height() + 0.15, f"{h.get_height():.5f}", color='black', ha='center')

def calc_means(d_exec):
    y = []
    names = []
    for j, c in enumerate(d_exec):
        if j in ignore_ixs:
            continue
            
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
    
    width = 0.3

    x = np.arange(len(d["exec"]) - len(ignore_ixs))
    y, names = calc_means(d["exec"])
    y4, names4 = calc_means(data_4[i]["exec"])
    
    draw_bars(ax, x, y, y4)

    _max = np.max(y)
    ax.set_ylim(0, _max + 0.2*_max)

    plt.xticks(x+width/2, ["Original (No Vec)", "CTB #1 (Vec) 1,4 threads", "CTB #2 (Vec) 1,4 threads"])

    legend += "\n"

    ax.set_title(script_args)
    ax.set_ylabel("(Mean) Seconds")

plt.tight_layout()

plt.savefig(exp_folder + "bars.png")
# plt.savefig("my_plot.svg", format='svg')

with open(exp_folder + "legends.txt", "w") as f:
    f.write(legend)