import os
import sys
import numpy as np
import subprocess
import json
import datetime
import argparse
from tabulate import tabulate
import pandas as pd

def extract_f1xm3(stdout):
    lines = stdout.split('\n')

    for line in reversed(lines):
        if "F1XM3" in line:
            number = line.split("F1XM3:")[1]
            return int(number)
    return None

def extract_papi(stdout):
    lines = stdout.split('\n')

    offset = 0
    for i, line in enumerate(lines):
        if line.startswith("PAPI-HL Output:"):
           offset = i
           break
    t = "\n".join(lines[offset+1:])
    j = json.loads(t)
    out = j["threads"]["0"]["regions"]["0"]
    del out["name"]
    del out["parent_region_id"]
    return out

#------------------------------------------------------------------------------
# GLOBAL
#------------------------------------------------------------------------------

TOOLS = {
    "PAPI_STD": {
        "ENV": {
            "PAPI_EVENTS": "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE-MISSES,perf::BRANCHES,perf::BRANCH-MISSES",
            "PAPI_REPORT": "1"
        },
        "START_OP": "startProfiling();",
        "STOP_OP": "stopProfiling();",
        "END_OP": "",
        "GET_INFO": extract_papi
    },
    "PAPI_L1": {
        "ENV": {
            "PAPI_EVENTS": "perf::L1-dcache-load-misses,perf::L1-dcache-loads,perf::L1-dcache-prefetches,perf::L1-icache-load-misses,perf::L1-icache-loads",
            "PAPI_REPORT": "1",
        },
        "START_OP": "startProfiling();",
        "STOP_OP": "stopProfiling();",
        "END_OP": "",
        "GET_INFO": extract_papi
    },
    "PAPI_MPLX": {
        "ENV": {
            "PAPI_EVENTS": "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE-MISSES,perf::BRANCHES,perf::BRANCH-MISSES,perf::L1-dcache-load-misses,perf::L1-dcache-loads,perf::L1-dcache-prefetches,perf::L1-icache-load-misses,perf::L1-icache-loads",
            "PAPI_REPORT": "1",
            "PAPI_MULTIPLEX": "1",
        },
        "START_OP": "startProfiling();",
        "STOP_OP": "stopProfiling();",
        "END_OP": "",
        "GET_INFO": extract_papi
    },
    "NOW": {
        "ENV": {},
        "START_OP": "start = now();",
        "STOP_OP": "end = now();",
        "END_OP": "print(\"F1XM3:\"+ (end - start));",
        "GET_INFO": extract_f1xm3
    }
}

BASE_COMMANDS = {
    "daphne-X86-64-org-bin": [
        ["./run-daphne.sh"],
        ["./run-daphne.sh", "--vec"]
    ],
    "daphne-X86-64-vec-bin": [
        ["./run-daphne.sh", "--vec", "--vec-type=GREEDY_1"],
        ["./run-daphne.sh", "--vec", "--vec-type=GREEDY_1", "--gr1-col"],
        ["./run-daphne.sh", "--vec", "--vec-type=GREEDY_2"]
    ]
}

EXPERIMENTS = {
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
        
    output = subprocess.run(["neofetch", "--stdout"], capture_output=True, text=True)
    with open(folder+"/neofetch.txt", "w") as f:
        f.write(output.stdout)
    
    return

def prepare_script(path, tool):
    lines = []
    with open(path, "r") as f:
        lines = f.readlines()

    with open("_bench.daph", "w") as file:
        for line in lines:
            if "<start>" in line:
                line = line.replace("<start>", TOOLS[tool]["START_OP"])
            if "<stop>" in line:
                line = line.replace("<stop>", TOOLS[tool]["STOP_OP"])
            if "<end>" in line:
                line = line.replace("<end>", TOOLS[tool]["END_OP"])
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
    
    items = EXPERIMENTS.items()
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


def run_command(command, cwd, env):
    _command = []
    _command += command 

    process = subprocess.Popen(_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env={**env, **os.environ})
    stdout, stderr = process.communicate()

    return stdout.decode(), stderr.decode()

#------------------------------------------------------------------------------
# ARGS
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument("--tool", type=str, choices=list(TOOLS.keys()), help="", required=True)
parser.add_argument("--exp", type=str, choices=list(EXPERIMENTS.keys()) + ["ALL"], default="ALL", help="")
parser.add_argument("--samples", type=int, default=3, help="")
parser.add_argument("--threads", type=int, nargs="+", default=[1], help="")
parser.add_argument("--batchSizes", type=int, default=[0], nargs="+", help="")

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == "__main__":

    args = parser.parse_args()
    exp_start = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    os.mkdir(exp_start)
    save_sys_info(exp_start) 
         
    experiments = generate_experiments(args.threads, args.batchSizes, args.exp)
    for i, e in enumerate(experiments):

        tool_env = TOOLS[args.tool]["ENV"]
        env_str = " ".join(f"{k}=\"{v}\"" for k, v in tool_env.items())

        prepare_script(e["script"]["path"], args.tool)

        for j, c in enumerate(e["exec"]):

            cmd = c["cmd"]
            cwd = c["cwd"]

            command_str = " ".join(cmd)
            
            name = e["script"]["name"]
            print(f"Run: {env_str} {command_str} ({name}) {cwd} {args.samples}")

            timings = []
            for _ in range(0, args.samples):
                stdout, stderr = run_command(cmd, cwd, tool_env)
                
                timing = json.loads(stderr)
                timing["tool"] = TOOLS[args.tool]["GET_INFO"](stdout)

                df = pd.json_normalize(timing, sep=".")
                print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
                timings.append(timing)
            
            experiments[i]["exec"][j]["timings"] = timings

    print()

    with open(exp_start + "/timings.json", "w+") as f:
        json.dump(experiments, f, indent=4)
        f.close()
