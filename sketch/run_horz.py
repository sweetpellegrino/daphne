import os
import sys
import numpy as np

to_print = False
if "--print" in sys.argv:
    to_print = True

max_horz_ops=50
min_horz_ops=2

num_points = 16
log = np.logspace(np.log10(min_horz_ops), np.log10(max_horz_ops), num=num_points)

#depth = int(sys.argv[1])
#width = int(sys.argv[2])

def generate_script(num_ops):

    script = []
    
    script.append("m0 = fill(1.0, 10000, 10000);")

    count = 0

    script.append("")
    for j in range(0, num_ops):
        script.append("v"+str(count)+" = sqrt(m0);")
        #script.append("v"+str(count)+" = sum(m0);") 
        count = count + 1
    script.append("")


    for j in range(0, count):
        script.append("print(v"+str(j)+"[0,0]);")

    #script.insert(0, "#total: "+str(width*depth+num_input+width))
    return script
    
command = ["../run-daphne.sh", "--vec", "--vec-type=GREEDY_1", "_horz.daph"]

for ops in log:

    print("Run: " + " ".join(command) + " " + str(int((ops))))
    print("Run:" + str(int((ops))))

    script = generate_script(int(ops))
    #with open("_horz.daph", "w") as f:
    #    for line in script:
    #        f.write(line + '\n')
    
    print(script)