import os
import json
import glob
import matplotlib.pyplot as plt

exp='csc_csr_sym'
result_dir =  'results/'+exp
folders = ["csr_aggCol","csc_aggCol","csr_aggRow","csc_aggRow"]

files_per_param = {}
for f in folders:
    files_per_param[f] = glob.glob(os.path.join(result_dir+"/"+f, '**/*.json'), recursive=True)

values = {}
for key, value in files_per_param.items():
    aggregate = {
           "cycles": [],
           "real_time_nsec": [],
           "perf::CYCLES": [],
           "perf::INSTRUCTIONS": [],
           "perf::CACHE-REFERENCES": [],
           "perf::CACHE-MISSES": [],
           "perf::BRANCHES": []
     }
    for file in value:
        with open(file, 'r') as f:
            data = json.load(f)
            for m in aggregate:
                aggregate[m].append(int(data['threads']['0']['regions']['0'][m]))

    values[key] = aggregate
print(values)

# Generate boxplots
fig, axes = plt.subplots(nrows=len(folders), ncols=1, figsize=(8, 12))

for i, (key, value) in enumerate(values.items()):
    agg_names = list(value.keys())
    metrics = [value[m] for m in agg_names]
    ax = axes[i]
    ax.boxplot(metrics, agg_names)
    ax.set_title(key)
    ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig(f'{exp}_boxplot.png', dpi=300)
