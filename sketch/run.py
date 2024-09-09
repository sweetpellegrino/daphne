import subprocess

def run_command(command): 

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode()


n = 94770

for i in range(n):
    print("--------------------------------------------------------------------")
    #command = ["../bin/daphne", "--vec", "--vec-type", "ONE", "--run-key", str(i), "--num-threads=1", "n_matrix_multi_complex_connected_leafs.daph"]
    command = ["../bin/daphne", "--vec", "--vec-type", "ONE", "--run-key", str(i), "--num-threads=1", "../test/api/cli/algorithms/kmeans.daphne", "r=1", "f=1", "c=1", "i=1"]
    _command = " ".join(command)
    print("command: " + _command)
    stdout, stderr = run_command(command)
    print(stdout)
    print(stderr)