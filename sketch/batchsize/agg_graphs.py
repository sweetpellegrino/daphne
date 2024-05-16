import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

result_dir =  'results'
sub = ["long", "sym", "wide"]
folders = ["csrcsr-aggCol","csccsc-aggCol","csrcsr-aggRow","csccsc-aggRow"]

ns = [name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))]

files_per_param = {}
for n in ns:
    for s in sub:
        exp=result_dir+"/"+n+"/"+s+"-"+n
        files_per_param[exp] = {}
        for f in folders:
            files_per_param[exp][f] = glob.glob(os.path.join(exp+"/"+f, '**/*.json'), recursive=True)

values = {}
for key, value in files_per_param.items():
    values[key] = {}
    for key2, value2 in value.items():
        aggregate = {
            "cycles": [],
            "real_time_nsec": [],
            "perf::CYCLES": [],
            "perf::INSTRUCTIONS": [],
            "perf::CACHE-REFERENCES": [],
            "perf::CACHE-MISSES": [],
            "perf::BRANCHES": []
        }
        for file in value2:
            with open(file, 'r') as f:
                data = json.load(f)
                for m in aggregate:
                    aggregate[m].append(int(data['threads']['0']['regions']['0'][m]))
        values[key][key2] = aggregate

moi = ["real_time_nsec", "perf::CACHE-MISSES"]
unit = ["ns", "# num"]
for i, (key, value) in enumerate(values.items()):
    print(key)
    fig, axes = plt.subplots(nrows=len(moi), ncols=1, figsize=(8, 12))
    _moi = {}
    for m in moi:
        _moi[m] = {}
        for j, (key2, value2) in enumerate(value.items()):
            _moi[m][key2] = value2[m]    

    for j, (key3, value3) in enumerate(_moi.items()):

        agg_names = value3.keys()
        metrics = list(value3.values())
        metrics = [sorted(metrics[i]) for i in range(0, len(metrics))]
 
        df = pd.DataFrame(np.array(metrics).T, columns=agg_names)
        ax1 = axes[j]
        df.boxplot(column=list(agg_names), grid=True, ax=ax1)
        #ax1.set_ylim(ymin=0)
        ax1.set_title(f'{key3}')
        ax1.set_xlabel(f'x data')
        ax1.set_ylabel(f'{unit[j]}')

    plt.tight_layout()
    plt.savefig(f'{key}/boxplot.png', dpi=300)