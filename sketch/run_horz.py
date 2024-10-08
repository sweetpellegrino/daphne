import os
import sys
import numpy as np
import subprocess
import json
import datetime

to_print = False
if "--print" in sys.argv:
    to_print = True

max_horz_ops=50
min_horz_ops=2

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
    
#num_points = 16
#log = np.logspace(np.log10(min_horz_ops), np.log10(max_horz_ops), num=num_points)

#depth = int(sys.argv[1])
#width = int(sys.argv[2])


prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def generate_script(num_ops):

    script = []
    
    script.append("X = fill(1.0, 5000, 5000);")
    script.append("start = now();")

    count = 0

    for j in range(0, num_ops):
        #script.append("v"+str(count)+" = "+ operators[j])
        script.append(generate_operator(j, "X"))
        count = count + 1
    script.append("end = now();")


    for j in range(0, count):
        script.append("print(v"+str(j)+"[0,0]);")

    script.append("print(\"F1XM3:\"+ (end - start));")

    #script.insert(0, "#total: "+str(width*depth+num_input+width))
    return script
    

def extract_f1xm3(stdout):
    lines = stdout.split('\n')

    for line in reversed(lines):
        if 'F1XM3' in line:
            number = line.split('F1XM3:')[1]
            return int(number)
    return None


#command = ["../bin/daphne", "--vec-type=GREEDY_1", "--num-threads=1", "_horz.daph"]
#command = ["../bin/daphne", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "_horz.daph"]

cwd = "daphne-X86-64-vec-bin"
commands = [
    ["./run-daphne.sh", "--timing", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "../_horz.daph"],
    ["./run-daphne.sh", "--timing", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "../_horz.daph"]
]

samples = 0

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

    process = subprocess.Popen(_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    stdout, stderr = process.communicate()

    return stdout.decode(), stderr.decode()

#for ops in log:

output = []
for c in commands: 

    print(c)

    _out = {}
    for ops in range(0, len(operators)):
    #for ops in range(0, 14):

        print("Run: " + " ".join(c) + " " + str(int((ops))))

        script = generate_script(int(ops))
        with open("_horz.daph", "w") as f:
            for line in script:
                f.write(line + '\n')

        timings = []
        for i in range(0, samples):
            stdout, stderr = run_command(c, cwd)
            
            timing = json.loads(stderr)
            timing["vectorized_nanoseconds"] = extract_f1xm3(stdout)

            print(timing)
            timings.append(timing)
       
        _out[ops] = timings 
        
    output.append({
        "cmd": c,
        "timings": _out
    })


with open(prefix + "-horz_timings.json", "w+") as f:
    json.dump(output, f, indent=4)
    f.close()