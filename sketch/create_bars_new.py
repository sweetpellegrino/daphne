import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

exp_folder = "results/ae-1-10/"
exp_folder_4 = "results/ae-48-10/"

ignore_ixs = []

for b in [True, False]:
    if b: 
        key = "tool"
        unit = 1e9
        unit_text = "(Mean) Execution Seconds"
        filename = "exec_perf"
    else: 
        key = "peak_rss_kilobytes"
        unit = 1e3
        unit_text = "(Max) Peak Resident Set Size"
        filename = "memory_perf"

    with open(exp_folder + "timings.json", "r") as f:
        data = json.load(f)
    with open(exp_folder_4 + "timings.json", "r") as f:
        data_4 = json.load(f)

    nrows = len(data)
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 4*nrows))
    plt.rcParams['font.size'] = 12
    width = 0.45

    # gray 
    # (245,245,245) : #f5f5f5
    # (102,102,102) : #666666 (0.40, 0.40, 0.40)
    # blue
    # (219,232,251) : #dbe8fb
    # (110,143,189) : #6e8fbd (0.43, 0.56, 0.74)
    # green
    # (214,232,213) : #d6e8d5
    # (131,178,106) : #83b26a (0.51, 0.70, 0.42)
    # orange
    # (255,230,206) : #ffe6ce
    # (214,154,35) : #d69a23 (0.84, 0.60, 0.14)
    # red
    # (247,206,206) : #f7cece
    # (182,85,82) : #b65552 (0.71, 0.33, 0.32)
    # purple
    # (225,214,231) :#e1d6e7
    # (150,116,165) :#9674a5 (0.59, 0.45, 0.65)

    #colors = ["tab:gray", "seagreen", "seagreen", "mediumseagreen", "mediumseagreen", "springgreen", "springgreen"]
    colors = ["#f5f5f5", "#dbe8fb", "#d6e8d5", "#e1d6e7", "#ffe6ce"]
    edgecolors = ["#666666", "#6e8fbd", "#83b26a", "#9674a5", "#d69a23"]

    def draw_bars(ax, x, y, y4):

        x2 = x + width
        x2[0] = x[0] 
        y4[0] = y[0]

        handle = ax.bar(x, y, width, color=colors, edgecolor=edgecolors)
        handle2 = ax.bar(x2, y4, width, color=edgecolors, edgecolor=edgecolors)

        if False:
            for i,h in enumerate(handle + handle2):
                if i == len(handle):
                    continue
                ax.text(h.get_x() + h.get_width() / 2.0, h.get_height() + 0.15, f"{h.get_height():.2f}", color='black', ha='center')

    def calc_means(d_exec, calc_max):
        y = []
        names = []
        for j, c in enumerate(d_exec):
            if j in ignore_ixs:
                continue
                
            timings = pd.DataFrame(c["timings"])
            if calc_max:
                mean = timings[key].max() / unit
            else:
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
        x[0] = x[0] + width/2 

        y, names = calc_means(d["exec"], False)
        y4, names4 = calc_means(data_4[i]["exec"], True)
        
        draw_bars(ax, x, y, y4)

        _max = np.max(y)
        _max4 = np.max(y4)
        if _max < _max4:
            _max = _max4
        
        ax.set_ylim(0, _max + 0.2*_max)

        x[0] = x[0] - width/2 
        plt.xticks(x+width/2, ["Base", "Base (Vec)", "CTB #1", "CTB #2", "CTB #3 "])

        #ax.set_title(script_args)
        ax.set_ylabel(unit_text)

    plt.tight_layout()

    plt.savefig(exp_folder + filename + ".png")
    plt.savefig(exp_folder + filename + ".svg", format='svg')