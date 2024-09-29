import subprocess
import json
import time
import os

#ld_lib = ["/root/daphne-org/lib", "/root/daphne-vec/lib"]
daphne_root = ["./daphne-org/run-daphne.sh", "./daphne-vec/run-daphne.sh"]
vec_type = [[], ["GREEDY_1", "GREEDY_2"]] #--vec-type
num_threads = ["1", "4"] #--num-threads
scripts = ["./sketch/kmeans.daphne"]

#generate commands
expand_ld_lib = []
commands = []
for root in daphne_root:
    for threads in num_threads:
        for script in scripts:
            if root == "./daphne-org/run-daphne.sh":
                l = [root, "--timing", "--num-threads="+threads, script]
                commands.append(l)
                l = [root, "--timing", "--vec", "--num-threads="+threads, script]
                commands.append(l)
            else:
                for vt in vec_type[1]:
                    #expand_ld_lib.append([ld_lib[1]]) 
                    l = [root, "--timing", "--vec", "--num-threads="+threads, "--vec-type="+vt, script]
                    commands.append(l)
                    

def run_command(command):
    #command = ["../bin/daphne","--timing", "--vec", "--vec-type", "ONE", "--run-key", str(i), "--num-threads=1", "../test/api/cli/algorithms/kmeans.daphne", "r=1", "f=1", "c=1", "i=1"]
    #_env = os.environ.copy()
    #_env["LD_LIBRARY_PATH"] = f"{ld_lib}:{_env['LD_LIBRARY_PATH']}"
    #_env["LD_LIBRARY_PATH"] = f"{ld_lib}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode()

samples = 3

for c in range(0, len(commands)):
    print("Running: " + " ".join(commands[c]) + " " + str(samples) + " times")
    for i in range(0, samples):
        stdout, stderr = run_command(commands[c])
        #stderr = json.dumps({"startup_seconds": 0.0343132, "parsing_seconds": 0.00495325, "compilation_seconds": 0.0732359, "execution_seconds": 59.493, "total_seconds": 59.6055})
        print(str(i) + ": " + stderr)