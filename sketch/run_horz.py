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
    t = "".join(lines[offset+1:])
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
            "PAPI_EVENTS": "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE MISSES,perf::BRANCHES,perf::BRANCH-MISSES",
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
            "PAPI_EVENTS": "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE MISSES,perf::BRANCHES,perf::BRANCH-MISSES,perf::L1-dcache-load-misses,perf::L1-dcache-loads,perf::L1-dcache-prefetches,perf::L1-icache-load-misses,perf::L1-icache-loads",
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

GENERATE_FUNCS = {
    "ADD": lambda i, arg: [f"v{i} = {arg} + {i * 0.1};"],
    "ADD_SUM": lambda i, arg: [f"i{i} = {arg} + {i * 0.1};", f"v{i} = sum(i{i});"]
}

GENERATE_PRINT_FUNCS = {
    "ADD": lambda i: [f"print(v{i}[0,0]);"],
    "ADD_SUM": lambda i: [f"print(v{i});"]
}

BASE_CWD = "daphne-X86-64-vec-bin"
BASE_COMMAND = lambda th, bs, no_hf: [
    "./run-daphne.sh",
    "--timing",
    "--vec",
    "--vec-type=GREEDY_1",
    f"--num-threads={th}",
    f"--batchSize={bs}",
    "--no-hf" if no_hf else "",
    "../_horz.daph"
]

#------------------------------------------------------------------------------
# HELPER
#------------------------------------------------------------------------------

def generate_script(num_ops, tool, func, rows, cols):

    script = []
    
    script.append(f"X = fill(1.0, {rows}, {cols});")
    script.append(TOOLS[tool]["START_OP"])

    for j in range(0, num_ops):
        script += GENERATE_FUNCS[func](j, "X")
    script.append(TOOLS[tool]["STOP_OP"])

    for j in range(0, num_ops):
        script += GENERATE_PRINT_FUNCS[func](j)

    script.append(TOOLS[tool]["END_OP"])

    return script


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
parser.add_argument("--tool", type=str, choices=TOOLS.keys(), help="", required=True)
parser.add_argument("--script", type=str, choices=GENERATE_FUNCS.keys(), help="", required=True)
parser.add_argument("--rows", type=int, default=10000, help="rows")
parser.add_argument("--cols", type=int, default=10000, help="rows")
parser.add_argument("--samples", type=int, default=3, help="")
parser.add_argument("--num-ops", type=int, default=12, help="")
parser.add_argument("--threads", type=int, default=1, help="")
parser.add_argument("--batchSize", type=int, default=0, help="")

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == "__main__":

    args = parser.parse_args()
    exp_start = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    output = []
    for no_hf in [False, True]: 
        
        tool_env = TOOLS[args.tool]["ENV"]
        command = BASE_COMMAND(args.threads, args.batchSize, no_hf)

        env_str = " ".join(f"{k}=\"{v}\"" for k, v in tool_env.items())
        command_str = " ".join(command)

        command_output = {}
        for ops in range(args.num_ops, args.num_ops+1):

            print(f"Run: {env_str} {command_str} {ops}")

            script = generate_script(ops, args.tool, args.script, args.rows, args.cols)
            with open("_horz.daph", "w") as f:
                for line in script:
                    f.write(line + '\n')

            timings = []
            for i in range(0, args.samples):
                stdout, stderr = run_command(command, BASE_CWD, tool_env)
                
                timing = json.loads(stderr)
                timing["tool"] = TOOLS[args.tool]["GET_INFO"](stdout)

                df = pd.json_normalize(timing, sep=".")
                print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
                timings.append(timing)
        
            command_output[ops] = timings 

            print()
            
        output.append({
            "cmd": command,
            "timings": command_output,
          
        })

    with open(exp_start + "-horz_timings.json", "w+") as f:
        _output = {
            "settings": {
                "num-ops": args.num_ops,
                "rows": args.rows,
                "cols": args.cols,
                "type": args.script,
                "tool": args.tool,
                "threads": args.threads,
                "samples": args.samples,
                "batchSize": args.batchSize
            }
            "execs": output
        }
        json.dump(_output, f, indent=4)
        f.close()