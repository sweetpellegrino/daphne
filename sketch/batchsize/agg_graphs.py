import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

result_dir =  'results/100000_100'
HEIGHT=100000
WIDTH=100

ns = [name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))]

files_per_param = {}
for n in ns:
    files_per_param[n] = glob.glob(os.path.join(result_dir+"/"+n, '**/*.json'), recursive=True)

values = {}
for key, value in files_per_param.items():
    values[key] = {}
    aggregate = {
        "cycles": [],
        "real_time_nsec": [],
        "perf::CYCLES": [],
        "perf::INSTRUCTIONS": [],
        "perf::CACHE-REFERENCES": [],
        "perf::CACHE-MISSES": [],
        "perf::BRANCHES": []
    }
    for value2 in value:
        with open(value2, 'r') as f:
            data = json.load(f)
            for m in aggregate:
                aggregate[m].append(int(data['threads']['0']['regions']['0'][m]))
    values[key] = aggregate

_keys = [int(k) for k in list(values.keys())]
_keys.sort()
values = {i: values[str(i)] for i in _keys}

moi = ["real_time_nsec", "perf::CACHE-MISSES"]
unit = ["ns", "# num"]

filter_key = 50

_moi = {}
for m in moi:
    _moi[m] = {}
    for j, (key2, value2) in enumerate(values.items()):
        if filter_key == -1 or filter_key > key2:
            _moi[m][key2] = value2[m]

print

_std_moi = {}
for key, values in _moi.items():
    _std_moi[key] = {}
    for key2, value2 in enumerate(values.items()):
        _std_moi[key][value2[0]] = np.std(value2[1])

mean_moi = {}
for key, value in _moi.items():
    mean_moi[key] = {k: round(sum(v)/len(v)) for k, v in value.items()}

'''
print("MAX")
print(max_moi[list(max_moi.keys())[0]])
print("MEAN")
print(mean_moi[list(mean_moi.keys())[0]])
print("MIN")
print(min_moi[list(min_moi.keys())[0]])
'''



fig, axes = plt.subplots(nrows=len(moi), ncols=1, figsize=(8, 12))
for j, key3 in enumerate(_moi.keys()):

        ax1 = axes[j]
        mean_lists = sorted(mean_moi[key3].items())
        std_lists = sorted(_std_moi[key3].items())
        x, y = zip(*mean_lists)
        _, e = zip(*std_lists)

        lower_bound = tuple(map(lambda i, j: i - j, y, e))
        upper_bound = tuple(map(lambda i, j: i + j, y, e))

        #ax1.plot(x, y, color="grey")
        ax1.plot(x, y, color="red")
        ax1.fill_between(x, lower_bound, upper_bound)
        #ax1.plot(x2, y3, color="grey")
        
        #ax1.set_ylim(ymin=0)
        ax1.set_title(f'{key3}')
        ax1.set_xlabel(f'x data')
        ax1.set_ylabel(f'{unit[j]}')

plt.tight_layout()
plt.savefig(f'{result_dir}/result.png', dpi=300)


'''
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
'''