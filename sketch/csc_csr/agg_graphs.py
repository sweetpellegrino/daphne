import os
import json
import glob
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
            print(exp)
            print(f)
            files_per_param[exp][f] = glob.glob(os.path.join(exp+"/"+f, '**/*.json'), recursive=True)

values = {}
for key, value in files_per_param.items():
    print(key)
    values[key] = {}
    for key2, value2 in value.items():
        print(key2)
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
for i, (key, value) in enumerate(values.items()):
    fig, axes = plt.subplots(nrows=len(moi), ncols=1, figsize=(8, 12))
    _moi = {}
    for m in moi:
        _moi[m] = {}
        for j, (key2, value2) in enumerate(value.items()):
            _moi[m][key2] = value2[m]    

    for j, (key3, value3) in enumerate(_moi.items()):
        print(j, key3, value3)
        agg_names = key3
        metrics = list(value3.values())
        metrics = [sorted(metrics[i]) for i in range(0, len(metrics))]
        print(metrics)
        ax = axes[j]
        ax.boxplot(metrics, agg_names)
        ax.set_title(f'{key3}')
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(f'{key}/boxplot.png', dpi=300)
