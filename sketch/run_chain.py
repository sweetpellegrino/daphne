import sys
import numpy as np
import json
import datetime
import argparse
from tabulate import tabulate
import pandas as pd
import shared as sh

#------------------------------------------------------------------------------
# GLOBAL
#------------------------------------------------------------------------------

GENERATE_FUNCS = {
    "T": lambda i, arg: [f"v{i} = t({arg});"],
    "ADD": lambda i, arg: [f"v{i} = {arg} + {i * 0.1 + 0.1};"],
}

GENERATE_PRINT_FUNCS = {
    "T": lambda i: [f"print(v{i}[0,0]);"],
    "ADD": lambda i: [f"print(v{i}[0,0]);"]
}

BASE_CWD = "daphne-X86-64-vec-bin"
GLOBAL_ARGS = []
BASE_COMMAND = lambda th, bs, gr1_col: [
    "./run-daphne.sh",
    "--timing",
    "--vec",
    "--vec-type=GREEDY_1",
    f"--num-threads={th}",
    f"--batchSize={bs}",
] + (["--gr1-col"] if gr1_col else []) + GLOBAL_ARGS + ["../_chain.daph"]

#------------------------------------------------------------------------------
# HELPER
#------------------------------------------------------------------------------

def generate_script(num_ops, tool, func, rows, cols):

    script = []
    
    script.append(f"X = fill(1.0, {rows}, {cols});")
    script.append(sh.TOOLS[tool]["START_OP"])

    for j in range(0, num_ops):
        if j == 0:
            script += GENERATE_FUNCS[func](j, "X")
        else:
            script += GENERATE_FUNCS[func](j, f"v{j - 1}")
    script.append(sh.TOOLS[tool]["STOP_OP"])

    script += GENERATE_PRINT_FUNCS[func](num_ops - 1)

    script.append(sh.TOOLS[tool]["END_OP"])

    return script

#------------------------------------------------------------------------------
# ARGS
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument("--tool", type=str, choices=sh.TOOLS.keys(), help="", required=True)
parser.add_argument("--script", type=str, choices=GENERATE_FUNCS.keys(), default="T", help="")
parser.add_argument("--rows", type=int, default=10000, help="rows")
parser.add_argument("--cols", type=int, default=10000, help="rows")
parser.add_argument("--samples", type=int, default=3, help="")
parser.add_argument("--num-ops", type=int, default=12, help="")
parser.add_argument("--threads", type=int, default=1, help="")
parser.add_argument("--batchSize", type=int, default=0, help="")
parser.add_argument("--verbose-output", action="store_true")
parser.add_argument("--explain", action="store_true")

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == "__main__":

    args = parser.parse_args()
    exp_start = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    if args.explain:
        GLOBAL_ARGS += ["--explain=vectorized"]

    output = []
    for gr1_col in [False, True]: 
        
        cmd = BASE_COMMAND(args.threads, args.batchSize, gr1_col)

        command_output = {}
        for ops in range(args.num_ops, args.num_ops+1):

            script = generate_script(ops, args.tool, args.script, args.rows, args.cols)
            with open("_chain.daph", "w") as f:
                for line in script:
                    f.write(line + '\n')

            timings = sh.runner(args, cmd, BASE_CWD) 
        
            #command_output[ops] = timings 
            command_output = timings 

        output.append({
            "cmd": cmd,
            "timings": command_output,
        })

    with open(exp_start + "-chain-timings.json", "w+") as f:
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
            },
            "execs": output
        }
        json.dump(_output, f, indent=4)
        f.close()
    
    for i in output:
        print(" ".join(i["cmd"]))
        df = pd.json_normalize(i["timings"], sep=".")
        tools_cols = [col for col in df.columns if col.startswith("tool")]
        df[tools_cols] = df[tools_cols].astype(int)
        print(tabulate(df.describe(), headers="keys", tablefmt="psql", showindex=True))