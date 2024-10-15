import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

exp_folder = "results/2024-10-05_19:58:15/"
with open(exp_folder + "timings.json", "r") as f:
    data = json.load(f)

nrows = len(data)
if len(data) == 1:
    nrows = 2

fig, axs = plt.subplots(nrows, 1, figsize=(12, 3*nrows))

colors = ["tab:gray", "seagreen", "seagreen", "mediumseagreen", "mediumseagreen", "springgreen", "springgreen"]

legend = ""

for i, d in enumerate(data):
    script_args = " ".join([d["script"]["path"]] + d["script"]["args"])

    ax = axs[i]
    x = []
    y = []
    legend += script_args + "\n"
    for j, c in enumerate(d["exec"]):
        timings = pd.DataFrame(c["timings"])
        mean = timings["vectorized_nanoseconds"].mean() / 1e9
        std = timings["vectorized_nanoseconds"].std() / 1e9

        name = "".join([s[-1] for s in c["cmd"]])
        legend += name + ": " + " ".join(c["cmd"]) + " of " + c["cwd"] + "\n"

        x.append(name)
        y.append(mean)

    handle = ax.bar(x, y, color=colors)
    for h in handle:
        height = h.get_height()
        ax.text(h.get_x() + h.get_width() / 2.0, h.get_height() + 0.15, f"{h.get_height():.5f}", color='black', ha='center')

    _max = np.max(y)
    ax.set_ylim(0, _max + 0.15*_max)

    legend += "\n"

    ax.set_title(script_args)
    ax.set_ylabel("(Mean) Seconds")
    ax.set_xlabel("Commands")

plt.tight_layout()

plt.savefig(exp_folder + "bars.png")
with open(exp_folder + "legends.txt", "w") as f:
    f.write(legend)