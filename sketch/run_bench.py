import os
import subprocess
import json
import datetime
import argparse
import shared as sh

#------------------------------------------------------------------------------
# GLOBAL
#------------------------------------------------------------------------------
RESULT_DIR = "results/"

BASE_COMMANDS = {
    "daphne-org": [
        ["./bin/daphne"],
        ["./bin/daphne", "--vec"]
    ],
    "daphne": [
        ["./bin/daphne", "--vec", "--vec-type=GREEDY_1"],
        ["./bin/daphne", "--vec", "--vec-type=GREEDY_2"],
        ["./bin/daphne", "--vec", "--vec-type=GREEDY_3"]
    ]
}

'''
BASE_COMMANDS = {
    "daphne-X86-64-org-bin": [
        ["./run-daphne.sh"],
        ["./run-daphne.sh", "--vec"]
    ],
    "daphne-X86-64-vec-bin": [
        ["./run-daphne.sh", "--vec", "--vec-type=GREEDY_1"],
        ["./run-daphne.sh", "--vec", "--vec-type=GREEDY_2"],
        ["./run-daphne.sh", "--vec", "--vec-type=GREEDY_3"]
    ]
}
'''

EXPERIMENTS = {
     "r": {
        "path": "./sketch/bench/running_example.daph",
        "args": ["r=30000", "c=30000"]
    },
    "ta": {
        "path": "./sketch/bench/rhs_t_add.daph",
        "args": ["r=30000", "c=30000"]
    },
    "ot": {
        "path": "./sketch/bench/outerAdd_t.daph",
        "args": ["r=30000", "c=30000"]
    },
    "oe": {
        "path": "./sketch/bench/outerAdd_exp.daph",
        "args": ["r=30000", "c=30000"]
    },
    "ote": {
        "path": "./sketch/bench/outerAdd_t_exp.daph",
        "args": ["r=30000", "c=30000"]
    },
    "ate": {
        "path": "./sketch/bench/abs_t_exp.daph",
        "args": ["r=30000", "c=30000"]
    },
    "kmeans": {
        "path": "./sketch/bench/kmeans.daphne",
        "args": ["r=1000000", "f=100", "c=500", "i=2"]
    },
    "components": {
        "path": "./sketch/bench/components.daphne",
        "args": ["n=1000", "e=750"]
    }
}

GLOBAL_ARGS = ["--timing"]

#------------------------------------------------------------------------------
# HELPER
#------------------------------------------------------------------------------

def save_sys_info(folder):
    output = subprocess.run(["lscpu"], capture_output=True, text=True)
    with open(folder+"/lscpu.txt", "w") as f:
        f.write(output.stdout)
        
    '''
    output = subprocess.run(["neofetch", "--stdout"], capture_output=True, text=True)
    with open(folder+"/neofetch.txt", "w") as f:
        f.write(output.stdout)
    '''

    return

def prepare_script(path, tool):
    lines = []
    with open(path, "r") as f:
        lines = f.readlines()

    with open("_bench.daph", "w") as file:
        for line in lines:
            if "<start>" in line:
                line = line.replace("<start>", sh.TOOLS[tool]["START_OP"])
            if "<stop>" in line:
                line = line.replace("<stop>", sh.TOOLS[tool]["STOP_OP"])
            if "<end>" in line:
                line = line.replace("<end>", sh.TOOLS[tool]["END_OP"])
            file.write(line)

def generate_experiments(ths, bss, exp):

    vec_commands = []
    for cwd, varis in BASE_COMMANDS.items():
        for var in varis:
            if "--vec" in var:
                for thread in ths:
                    vec_commands.append({
                        "cwd": cwd,
                        "cmd": var + ["--num-threads="+ str(thread)] + GLOBAL_ARGS
                    })
            else:
                vec_commands.append({
                    "cwd": cwd,
                    "cmd": var + GLOBAL_ARGS
                })

    batch_commands = []
    for i, item in enumerate(vec_commands):
        for bs in bss:
            batch_commands.append({
                "cwd": cwd,
                "cmd": item["cmd"] + ["--batchSize="+str(bs)]
            })
    
    if not batch_commands:
        batch_commands = vec_commands

    experiments = []
    
    items = EXPERIMENTS
    if exp != "ALL":
        items = {
            args.exp: EXPERIMENTS[args.exp]
        }

    for k, v in items.items():
        _commands = []
        for item in vec_commands:
            cwd = item["cwd"]
            cmd = item["cmd"]
            command = {
                "cwd": cwd,
                "cmd": cmd + ["../_bench.daph"] + v["args"]
            }
            _commands.append(command)
        v["name"] = k
        experiments.append({
            "script": v,
            "exec": _commands
        })
    
    return experiments

#------------------------------------------------------------------------------
# ARGS
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument("--tool", type=str, choices=list(sh.TOOLS.keys()), help="", required=True)
parser.add_argument("--exp", type=str, choices=list(EXPERIMENTS.keys()) + ["ALL"], default="ALL", help="")
parser.add_argument("--samples", type=int, default=3, help="")
parser.add_argument("--threads", type=int, nargs="+", default=[1], help="")
parser.add_argument("--batchSizes", type=int, default=[0], nargs="+", help="")
parser.add_argument("--verbose-output", action="store_true")
parser.add_argument("--explain", action="store_true")

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == "__main__":

    args = parser.parse_args()
    exp_start = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    os.mkdir(exp_start)
    save_sys_info(exp_start) 

    if args.explain:
        GLOBAL_ARGS += ["--explain=vectorized"]
         
    experiments = generate_experiments(args.threads, args.batchSizes, args.exp)
    for i, e in enumerate(experiments):

        name = e["script"]["name"]
        print(f"Preparing script: {name}")

        prepare_script(e["script"]["path"], args.tool)

        for j, c in enumerate(e["exec"]):

            cmd = c["cmd"]
            cwd = c["cwd"]

            timings = sh.runner(args, cmd, cwd) 
            
            experiments[i]["exec"][j]["timings"] = timings

    print()

    with open(RESULT_DIR + exp_start + "/timings.json", "w+") as f:
        json.dump(experiments, f, indent=4)
        f.close()
