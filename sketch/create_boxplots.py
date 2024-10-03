import matplotlib.pyplot as plt
import numpy as np
import json
from textwrap import wrap

folder = "2024-10-02_10:52:02"

commands_json = folder + "/commands.json"

with open(commands_json, 'r') as f:
    commands = json.load(f)

command_timings = []
for i in range(0, len(commands)):
    command = commands[i]
    timings_json = folder + "/" + str(i) + "/timings.json"

    with open(timings_json, 'r') as f:
        timings = json.load(f)

    command_timings.append(
        {
            "cmd": command,
            "timings": timings
        }
    )

fig, axs = plt.subplots(len(commands), 1, figsize=(7, 5*len(commands)))

for i, ax in enumerate(fig.axes):
    items = [timings["execution_seconds"] for timings in command_timings[i]["timings"]]

    median = np.median(items)
    ax.boxplot(items)

    #ax.text(median, i*5, median, color='red', ha='center')

    ax.set_title("\n".join(wrap(" ".join(command_timings[i]["cmd"]["cmd"]))))
    ax.set_ylim([0, max(items) + 0.25])
    ax.set_xlabel("s")

plt.tight_layout()
fig.savefig('bp.png', dpi=300)