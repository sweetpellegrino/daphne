import os
import sys
import numpy as np
import subprocess

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

#num_points = 16
#log = np.logspace(np.log10(min_horz_ops), np.log10(max_horz_ops), num=num_points)

#depth = int(sys.argv[1])
#width = int(sys.argv[2])

def generate_script(num_ops):

    script = []
    
    script.append("X = fill(1.0, 5000, 5000);")
    script.append("start = now();")

    count = 0

    for j in range(0, num_ops):
        script.append("v"+str(count)+" = "+ operators[j])
        #script.append("v"+str(count)+" = sqrt(m0);")
        #script.append("v"+str(count)+" = sum(m0);") 
        count = count + 1
    script.append("end = now();")


    for j in range(0, count):
        script.append("print(v"+str(j)+"[0,0]);")

    script.append("print(\"F1XM3:\"+ (end - start));")

    #script.insert(0, "#total: "+str(width*depth+num_input+width))
    return script
    
#command = ["../bin/daphne", "--vec-type=GREEDY_1", "--num-threads=1", "_horz.daph"]
#command = ["../bin/daphne", "--vec", "--no-hf", "--vec-type=GREEDY_1", "--num-threads=1", "_horz.daph"]
command = ["../bin/daphne", "--vec", "--vec-type=GREEDY_1", "--num-threads=1", "_horz.daph"]

def run_command(command, cwd):
    _command = []
    _command += command 

    process = subprocess.Popen(_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    stdout, stderr = process.communicate()

    return stdout.decode(), stderr.decode()

#for ops in log:
#for ops in range(0, len(operators)):
for ops in range(0, 27):

    print("Run: " + " ".join(command) + " " + str(int((ops))))

    script = generate_script(int(ops))
    with open("_horz.daph", "w") as f:
        for line in script:
            f.write(line + '\n')

    stdout, stderr = run_command(command, "./");

    print(stdout)
    print(stderr)

    
    