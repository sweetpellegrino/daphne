from os import listdir
from os.path import isfile, join
import json
import pandas as pd
import matplotlib.pyplot as plt

d = "results/horz_add_sum/"

files = [f for f in listdir(d) if isfile(join(d, f))]

timings_hf = {}
timings_no_hf = {}
for file in files:
    if file[-3:] == "png":
        continue

    with open(d + file, "r") as f:
        data = json.load(f)
        num_ops = data["settings"]["num-ops"]
        _timings = data["execs"][0]["timings"]
        
        _t_df = pd.DataFrame(_timings)
        mean_time = _t_df["tool"].mean() / 1e9
        peak_mem = _t_df["peak_rss_kilobytes"].max() / 1e3

        timings_hf[num_ops] = {
            "tool" : mean_time,
            "peak_rss_kilobytes" : peak_mem
        }

        _timings = data["execs"][1]["timings"]
        
        _t_df = pd.DataFrame(_timings)
        mean_time = _t_df["tool"].mean() / 1e9
        peak_mem = _t_df["peak_rss_kilobytes"].max() / 1e3

        timings_no_hf[num_ops] = {
            "tool" : mean_time,
            "peak_rss_kilobytes" : peak_mem
        }

x = list(timings_hf.keys())
x.sort()
y1 = [timings_hf[ops]["tool"] for ops in x]
y2 = [timings_no_hf[ops]["tool"] for ops in x]


fig, ax = plt.subplots()

ax.plot(x, y1, marker='o', color="b")
ax.plot(x, y2, marker='o', color="r")

plt.title("Horizontal fusion")  
plt.xlabel("Num operators")
plt.ylabel("Seconds / Megabytes")

fig.tight_layout()
plt.savefig(d + "lines.png") 


    
