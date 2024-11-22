import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plot_config as pc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches


exp_name = "ae"

exp_folder   = f"results/microbenchmark/{exp_name}/{exp_name}-1-10-papi/"
exp_folder_4 = f"results/microbenchmark/{exp_name}/{exp_name}-48-10-papi/"

for b in [True, False]:
    if b: 
        key = "tool.real_time_nsec"
        unit = pc.units["exec_time"]["conversion"]
        unit_text = pc.units["exec_time"]["label"]
        filename = f"{exp_name}-time"
    else: 
        key = "tool.perf::CACHE-REFERENCES"
        unit = 1e6
        unit_text = "(Mean) Cache References (LLC) in Million"
        filename = f"{exp_name}-cache"

    with open(exp_folder + "timings.json", "r") as f:
        data = json.load(f)
    with open(exp_folder_4 + "timings.json", "r") as f:
        data_4 = json.load(f)

    fig, axs = plt.subplots(1, 1, figsize=(pc.figsize_width, pc.figsize_height))
    plt.rcParams['font.size'] = pc.font_size

    legend_handles = [
        mpatches.Patch(facecolor=pc.colors[0], edgecolor=pc.edgecolors[0], label='Scalar'),
        mpatches.Patch(facecolor=pc.colors[0], edgecolor=pc.edgecolors[0], hatch='///', label='1 Thread'),
        mpatches.Patch(facecolor=pc.edgecolors[0], edgecolor=pc.edgecolors[0], label='48 Threads')
    ]

    plt.legend(handles=legend_handles)

    def draw_bars(ax, x, y, y4):

        x2 = x + pc.bar_width
        x2[0] = x[0] 
        y4[0] = 0

        handle = ax.bar(x, y, pc.bar_width, color=pc.colors, edgecolor=pc.edgecolors)
        handle2 = ax.bar(x2, y4, pc.bar_width, color=pc.edgecolors, edgecolor=pc.edgecolors)
        
        for i,h in enumerate(handle):
            if i != 0:
                h.set_hatch("/")

    def calc_means(d_exec):
        y = []
        names = []
        for j, c in enumerate(d_exec):
                
            timings = pd.json_normalize(c["timings"], sep=".") 
            timings = timings.astype({"tool.real_time_nsec": int, "tool.perf::CACHE-MISSES": int,  "tool.perf::CACHE-REFERENCES": int})
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
        y4, names4 = calc_means(data_4[i]["exec"])
        
        draw_bars(ax, x, y, y4)

        _max = np.max(y)
        _max4 = np.max(y4)
        if _max < _max4:
            _max = _max4
        
        ax.set_ylim(0, _max + pc.offset_max*_max)

        x[0] = x[0] - pc.bar_width/2 
        plt.xticks(x+pc.bar_width/2, pc.xticks_name)

        if not b:
            ax_inset = inset_axes(ax, width="45%", height="45%", loc="center right")
            handle = ax_inset.bar(x, y, pc.bar_width, color=pc.colors, edgecolor=pc.edgecolors)
            for i,h in enumerate(handle):
                if i != 0:
                    h.set_hatch("/")
            ax_inset.bar(x + pc.bar_width, y4, pc.bar_width, color=pc.edgecolors, edgecolor=pc.edgecolors)
            ax_inset.set_xticks(x+pc.bar_width/2)
            ax_inset.set_xticklabels([])
            ax_inset.set_xlim(1.70, 4.75)
            ax_inset.set_ylim(0, np.max(y4[2:]) + pc.offset_max*np.max(y4[2:]))

            mark_inset(ax, ax_inset, loc1=3, loc2=4)

        #ax.set_title(script_args)
        ax.set_ylabel(unit_text)

    plt.tight_layout(pad=0)
    plt.savefig(exp_folder + filename + ".svg", format='svg')