import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plot_config as pc
import matplotlib.patches as mpatches

exp_name = "tsate"
exp_folder = f"results/microbenchmark/{exp_name}/"

with open(exp_folder + "static-metrics.json", "r") as f:
    data = json.load(f)

fig, axs = plt.subplots(1, 1, figsize=(pc.figsize_width, pc.figsize_height))
plt.rcParams['font.size'] = pc.font_size

legend_handles = [
    mpatches.Patch(facecolor=pc.colors[0], edgecolor=pc.edgecolors[0], label=r"No. $\it{vect.}$ ops"),
    mpatches.Patch(facecolor=pc.edgecolors[0], edgecolor=pc.edgecolors[0], label="Mean pipe. size")
]
plt.legend(handles=legend_handles)

df = pd.DataFrame(data)
x = np.arange(len(df))

plt.barh(x, df["no-vec-ops"], pc.bar_width, color=pc.colors, edgecolor=pc.edgecolors)
plt.barh(x+pc.bar_width, df["mean-op-pipe"], pc.bar_width, color=pc.edgecolors, edgecolor=pc.edgecolors)

plt.xlabel("Number of operators")
axs.text(0.15, pc.bar_width/2+0.1, "No vectorization", fontsize=10)

plt.yticks(x+pc.bar_width/2, pc.xticks_name)
plt.xticks(np.arange(0, int(np.max(df)+1), 1))

plt.gca().invert_yaxis()

_max = np.max(df["mean-op-pipe"].max())
_max4 = np.max(df["no-vec-ops"].max())
if _max < _max4:
    _max = _max4

axs.set_xlim(0, _max + pc.offset_max*_max)

plt.tight_layout(pad=0)
plt.savefig(exp_folder + f"{exp_name}-static-metrics.svg", format='svg') 