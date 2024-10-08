import subprocess
import json
import time
import os
import sys
import re
import datetime

use_perf = False
if "--use-perf" in sys.argv:
    use_perf = True

to_print = False
if "--print" in sys.argv:
    to_print = True

#------------------------------------------------------------------------------
# EXPERIMENTS
#------------------------------------------------------------------------------

samples = 5

'''
scripts = [
    {
        "script": "../sketch/bench/single_sum.daph",
        "args": ["X=\"../m_200000_100.csv\""]
    },
    {
        "script": "../sketch/bench/transpose_sum.daph",
        "args": ["X=\"../m_200000_100.csv\""]
    },
    {
        "script": "../sketch/bench/transpose_chain.daph",
        "args": ["X=\"../m_200000_100.csv\""]
    },
     {
        "script": "../sketch/bench/complex_sum.daph",
        "args": ["X=\"../m_20000_1000.csv\"", "Y=\"../m_1000_20000.csv\"", "Z=\"../m2_20000_1000.csv\"",  "H=\"../m2_1000_20000.csv\""]
    },
    {
        "script": "../sketch/bench/kmeans.daphne",
        "args": ["X=\"../m_200000_100.csv\"", "C=\"../m_50_100.csv\"", "i=2"]
    },
    {
        "script": "../sketch/bench/lmDS_rnd.daphne",
        "args": ["r=100000", "c=500", "rep=2", "icpt=1"]
    },
]
'''

scripts = [
    {
        "path": "../sketch/bench/outerAdd_t.daph",
        "args": ["r=30000", "c=30000"]
    },
    {
        "path": "../sketch/bench/outerAdd_exp.daph",
        "args": ["r=30000", "c=30000"]
    },
    {
        "path": "../sketch/bench/outerAdd_t_exp.daph",
        "args": ["r=30000", "c=30000"]
    },
    {
        "path": "../sketch/bench/abs_t_exp.daph",
        "args": ["r=30000", "c=30000"]
    },
]
'''
    {
        "path": "../sketch/bench/sqrt_sum.daph",
        "args": ["r=30000", "c=30000"]
    },
    {
        "path": "../sketch/bench/transpose_sum.daph",
        "args": ["r=30000", "c=30000"]
    },
    {
        "path": "../sketch/bench/kmeans.daphne",
        "args": ["r=1000000", "f=100", "c=500", "i=2"]
    }
]
'''


num_threads = ["1", "4"]
global_args = ["--timing"]

#------------------------------------------------------------------------------
# HELPER
#------------------------------------------------------------------------------

def extract_perf_stats(input_string):

    return {
        "cycles":               re.search(r'(\d+)\W+cycles', input_string).group(1),
        "instructions":         re.search(r'(\d+)\W+instructions', input_string).group(1),
        "cache-references":     re.search(r'(\d+)\W+cache-references', input_string).group(1),
        "cache-misses":         re.search(r'(\d+)\W+cache-misses', input_string).group(1),
        "seconds time elapsed": re.search(r'(\d+\.\d+)\W+seconds time elapsed', input_string).group(1),
        "seconds user":         re.search(r'(\d+\.\d+)\W+seconds user', input_string).group(1),
        "seconds sys":          re.search(r'(\d+\.\d+)\W+seconds sys', input_string).group(1),
    }

def save_sys_info(folder):
    output = subprocess.run(["lscpu"], capture_output=True, text=True)
    with open(folder+"/lscpu.txt", "w") as f:
        f.write(output.stdout)
        
    output = subprocess.run(["neofetch", "--stdout"], capture_output=True, text=True)
    with open(folder+"/neofetch.txt", "w") as f:
        f.write(output.stdout)
    
    return

#------------------------------------------------------------------------------
def generate_commands():

    vec_settings = {
        "daphne-X86-64-org-bin": [
            ["./run-daphne.sh"],
            ["./run-daphne.sh", "--vec"]
        ],
        "daphne-X86-64-vec-bin": [
            ["./run-daphne.sh", "--vec", "--vec-type=GREEDY_1"],
            ["./run-daphne.sh", "--vec", "--vec-type=GREEDY_2"]
        ]
    }

    #commands
    vec_commands = []
    for cwd, varis in vec_settings.items():
        for var in varis:
            if "--vec" in var:
                for thread in num_threads:
                    vec_commands.append({
                        "cwd": cwd,
                        "cmd": var + ["--num-threads="+ str(thread)] + global_args
                    })
            else:
                vec_commands.append({
                    "cwd": cwd,
                    "cmd": var + global_args
                })

    experiments = []
    for script in scripts:
        _commands = []
        for item in vec_commands:
            cwd = item["cwd"]
            cmd = item["cmd"]
            command = {
                "cwd": cwd,
                "cmd": cmd + [script["path"]] + script["args"]
            }
            _commands.append(command)
        experiments.append({
            "script": script,
            "exec": _commands
        })
    
    return experiments

experiments = generate_commands()

'''
matrices = [
    {
        "seed": "",
        "cols": "",
        "rows": "",
    },
]
'''

exp_folder = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
def setup_exp_run():
    os.mkdir(exp_folder)

    save_sys_info(exp_folder) 

    '''
    with open(exp_folder + "/commands.json", "w+") as f:
        json.dump(commands, f, indent=4)
        f.close()

    if os.path.exists("done"):
        with open("done", "r+") as f:
            j = json.load(f)
            seeds = j["seeds"]
            seeds.append(seed)
    else:
        with open("done", "w+") as f:
            json.dump({"seeds": [seed]}, f, indent=4)
            f.close()
    
    if not os.path.exists("done"):
        with open("done", "w+") as f:
            pass
    '''
         
    return

def run_command(command, cwd):
    _command = []
    if use_perf:
        #probably change to internal
        raise Exception("not tested");
        command += ["perf", "stat", "-e", "cycles,instructions,cache-references,cache-misses"]
    _command += command 

    process = subprocess.Popen(_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    stdout, stderr = process.communicate()

    return stdout.decode(), stderr.decode()

def extract_f1xm3(stdout):
    lines = stdout.split('\n')

    for line in reversed(lines):
        if 'F1XM3' in line:
            number = line.split('F1XM3:')[1]
            return int(number)
    return None


if to_print:
    for i, e in enumerate(experiments):
        for j, c in enumerate(e["exec"]):
            print(str(i) + ": " + " ".join(c["cmd"]) + " in " + c["cwd"])

    exit(0)

setup_exp_run()

for i, e in enumerate(experiments):
 
    for j, c in enumerate(e["exec"]):

        cmd = c["cmd"]
        cwd = c["cwd"]
        
        print("Running: " + " ".join(cmd) + " in " + cwd + ", " + str(samples) + " times")

        timings = []
        for _ in range(0, samples):
            stdout, stderr = run_command(cmd, cwd)
            
            timing = json.loads(stderr)
            timing["vectorized_nanoseconds"] = extract_f1xm3(stdout)

            print(timing)
            timings.append(timing)
        
        experiments[i]["exec"][j]["timings"] = timings

with open(exp_folder + "/timings.json", "w+") as f:
    json.dump(experiments, f, indent=4)
    f.close()