import os
import sys
import numpy as np
import subprocess
import json
import datetime

to_print = False
if "--print" in sys.argv:
    to_print = True

max_horz_ops=250
min_horz_ops=1

_env = {
    "PAPI_EVENTS": "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE MISSES,perf::BRANCHES,perf::BRANCH-MISSES",
    #"PAPI_EVENTS": "perf::L1-dcache-load-misses,perf::L1-dcache-loads,perf::L1-dcache-prefetches,perf::L1-icache-load-misses,perf::L1-icache-loads",
    "PAPI_REPORT": "1"
}

operators = [
    "minus(X);",
    "abs(X);",
    "sign(X);",
    "exp(X);",
    "ln(X);",
    "mod(X, 2);",
    "sqrt(X);",
    "round(X);",
    "floor(X);",
    "ceil(X);",
    "sin(X);",
    "cos(X);",
    "tan(X);",
    "sinh(X);",
    "cosh(X);",
    "tanh(X);",
    "asin(X);",
    "acos(X);",
    "atan(X);",
    "isNan(X);",
    "pow(X, 2);",
    "log(X, 2);",
    "min(X, 2);",
    "max(X, 2);",
    "X + 0.5;",
    "X - 0.5;",
    "X * 0.5;",
    "X / 0.5;",
]

def generate_operator(i, arg):
    return "v" + str(i) + " = " + arg + " + " +  str(i*0.1) + ";"
    
num_points = 20
log = np.logspace(np.log10(min_horz_ops), np.log10(max_horz_ops), num=num_points)

#depth = int(sys.argv[1])
#width = int(sys.argv[2])


prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def generate_script(num_ops):

    script = []
    
    script.append("X = fill(1.0, 10000, 10000);")
    #script.append("start = now();")
    script.append("startProfiling();")

    count = 0

    for j in range(0, num_ops):
        #script.append("v"+str(count)+" = "+ operators[j])
        script.append(generate_operator(j, "X"))
        script.append("s"+str(j)+" = sum(v"+str(j)+");")
        count = count + 1
    script.append("stopProfiling();")
    #script.append("end = now();")

    for j in range(0, count):
        #script.append("print(v"+str(j)+"[0,0]);")
        script.append("print(s"+str(j)+");")

    #script.append("print(\"F1XM3:\"+ (end - start));")

    #script.insert(0, "#total: "+str(width*depth+num_input+width))
    return script
    

def extract_f1xm3(stdout):
    lines = stdout.split('\n')

    for line in reversed(lines):
        if 'F1XM3' in line:
            number = line.split('F1XM3:')[1]
            return int(number)
    return None

def extract_papi(stdout):
    lines = stdout.split('\n')

    offset = 0
    for i, line in enumerate(lines):
        if line.startswith('PAPI-HL Output:'):
           offset = i
           break
    t = "".join(lines[offset+1:])
    j = json.loads(t)
    return j['threads']["0"]["regions"]["0"]


#command = ["../bin/daphne", "--vec-type=GREEDY_1", "--num-threads=1", "_horz.daph"]
#command = ["../bin/daphne", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "_horz.daph"]

cwd = "daphne-X86-64-vec-bin"
commands = [["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize="+str(int(l)), "../_horz.daph"] for l in log]
'''
commands = [
    ["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=1", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=5", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=25", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=50", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=100", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=250", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=1", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=5", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=25", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=50", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=100", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "--batchSize=250", "../_horz.daph"],
]
'''

samples = 3

'''
cwd = "./"
commands = [
    ["../bin/daphne", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "./_horz.daph"],
    ["../bin/daphne", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "./_horz.daph"]
]
'''

def run_command(command, cwd):
    _command = []
    _command += command 

    process = subprocess.Popen(_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=_env)
    stdout, stderr = process.communicate()

    return stdout.decode(), stderr.decode()

#for ops in log:

output = []
for c in commands: 

    
    env_str = ""
    for k, v in _env.items():
        env_str += k +"=\""+v+"\" "

    _out = {}
    for ops in range(16, 17):
    #for ops in range(0, 14):

        print("Run: " + env_str + " ".join(c) + " " + str(int((ops))))

        script = generate_script(int(ops))
        with open("_horz.daph", "w") as f:
            for line in script:
                f.write(line + '\n')

        timings = []
        for i in range(0, samples):
            stdout, stderr = run_command(c, cwd)
            
            timing = json.loads(stderr)
            #timing["vectorized_nanoseconds"] = extract_f1xm3(stdout)
            timing["papi"] = extract_papi(stdout)

            print(timing)
            timings.append(timing)
       
        _out[ops] = timings 

        print()
        
    output.append({
        "cmd": c,
        "timings": _out
    })


with open(prefix + "-horz_timings.json", "w+") as f:
    json.dump(output, f, indent=4)
    f.close()
