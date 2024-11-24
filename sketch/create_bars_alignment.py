import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plot_config as pc
import matplotlib.patches as mpatches

exp_name = "ot"
exp_folder = f"results/microbenchmark/{exp_name}/"

with open(exp_folder + "alignment.json", "r") as f:
    data = json.load(f)

fig, axs = plt.subplots(1, 1, figsize=(pc.figsize_width, pc.figsize_height))
plt.rcParams['font.size'] = pc.font_size

legend_handles = [
    mpatches.Patch(facecolor=pc.colors[0], edgecolor=pc.edgecolors[0], label=r"Internal alignment"),
    mpatches.Patch(facecolor=pc.edgecolors[0], edgecolor=pc.edgecolors[0], label="Border alignment")
]
plt.legend(handles=legend_handles)

df = pd.DataFrame(data)
x = np.arange(len(df))

handle = plt.barh(x, df["internal-score"], pc.bar_width, color=pc.colors, edgecolor=pc.edgecolors)
handle = plt.barh(x+pc.bar_width, df["border-score"], pc.bar_width, color=pc.edgecolors, edgecolor=pc.edgecolors)

plt.xlabel("Alignment")
axs.text(0.05, pc.bar_width/2+0.1, "No vectorization", fontsize=10)

plt.yticks(x+pc.bar_width/2, pc.xticks_name)
#plt.xticks(x, x)

plt.gca().invert_yaxis()

axs.set_xticks([0, 0.25, 0.5, 0.75, 1])
axs.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
axs.set_xlim(0, 1.2)

plt.tight_layout(pad=0)
plt.savefig(exp_folder + f"{exp_name}-alignment.svg", format='svg') 