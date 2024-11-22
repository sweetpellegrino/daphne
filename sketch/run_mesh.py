import json
import datetime
import argparse
from tabulate import tabulate
import pandas as pd
import shared as sh

#------------------------------------------------------------------------------
# GLOBAL
#------------------------------------------------------------------------------
RESULT_DIR = "results/"
# For testing compilation time, we need this:
ROWS = 1
COLS = 1

BASE_CWD = "daphne"
GLOBAL_ARGS = []
BASE_COMMAND = lambda th, bs, vt: [
    "./bin/daphne",
    "--timing",
    "--vec",
    f"--vec-type={vt}",
    f"--num-threads={th}",
    f"--batchSize={bs}",
] + GLOBAL_ARGS + ["../_mesh.daph"]

#------------------------------------------------------------------------------
# HELPER
#------------------------------------------------------------------------------

def generate_script(tool, depth, width):

    script = []
    count = 0

    for i in range (0, width):
        script.append(f"m{i} = fill({i+1}, {ROWS}, {COLS});")

    script.append(sh.TOOLS[tool]["START_OP"])

    for i in range(0, depth):
        for j in range(0, width):
            if i == 0 and j == 0:
                script.append("i"+str(count)+" = sqrt(m"+str(count % width) +");")
            elif j == 0:
                script.append("i"+str(count)+" = sqrt(i"+str(count-width)+");")
            elif i == 0:
                script.append("i"+str(count)+" = m"+str((count+1) % width)+"+"+"m"+str((count+2) % width)+";")
            else:
                script.append("i"+str(count)+" = i"+str(count-width)+"+"+"i"+str(count-width+1)+";")

            count = count + 1
        script.append("")

    script.append(sh.TOOLS[tool]["STOP_OP"])

    for j in range(0, width):
        script.append("print(i"+str(count-1-j)+"[0,0]);")

    script.append(sh.TOOLS[tool]["END_OP"])

    script.insert(0, "#width: "+str(width) + " depth:" +str(depth))
    script.insert(0, "#vec: "+str(width*depth))
    script.insert(0, "#total: "+str(width*depth+depth+width))

    return script

#------------------------------------------------------------------------------
# ARGS
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument("--tool", type=str, choices=sh.TOOLS.keys(), default="NOW", help="")
parser.add_argument("--depth", type=int, default=10, help="depth")
parser.add_argument("--width", type=int, default=10, help="width")
parser.add_argument("--samples", type=int, default=3, help="")
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
    for vt in ["GREEDY_1", "GREEDY_2", "GREEDY_3"]: 
        
        cmd = BASE_COMMAND(args.threads, args.batchSize, vt)

        command_output = {}

        script = generate_script(args.tool, args.depth, args.width)
        with open("_mesh.daph", "w") as f:
            for line in script:
                f.write(line + '\n')

        timings = sh.runner(args, cmd, BASE_CWD) 

        #command_output[ops] = timings 
        command_output = timings 

        print()
            
        output.append({
            "cmd": cmd,
            "timings": command_output,
          
        })

    with open(RESULT_DIR + exp_start + "-mesh_timings.json", "w+") as f:
        _output = {
            "settings": {
                "rows": ROWS,
                "cols": COLS,
                "depth": args.depth,
                "width": args.width,
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

        