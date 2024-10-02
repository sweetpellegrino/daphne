import os
import sys

# okay sketch && python3 gen_net_scripts.py 25 25 && cd ..
# good sketch && python3 gen_net_scripts.py 10 25 && cd ..
# ./bin/daphne --vec --timing --vec-type GREEDY_2  ./sketch/net_50-25.daph 

depth = int(sys.argv[1])
width = int(sys.argv[2])

def generate_script():

    script = []
    count = 0
    
    num_input = 4
    script.append("m0 = fill(1.0, 10, 10);")
    script.append("m1 = fill(2.0, 10, 10);")
    script.append("m2 = fill(3.0, 10, 10);")
    script.append("m3 = fill(4.0, 10, 10);")

    script.append("")
    for i in range(0,depth):
        for j in range(0, width):
            if i == 0 and j == 0:
                script.append("v"+str(count)+" = sqrt(m"+str(count % num_input) +");")
            elif j == 0:
                script.append("v"+str(count)+" = sqrt(v"+str(count-width)+");")
            elif i == 0:
                script.append("v"+str(count)+" = m"+str((count+1) % num_input)+"+"+"m"+str((count+2) % num_input)+";")
            else:
                script.append("v"+str(count)+" = v"+str(count-width)+"+"+"v"+str(count-width+1)+";")

            count = count + 1
        script.append("")

    for j in range(0, width):
        script.append("print(v"+str(count-1-j)+"[0,0]);")

    script.insert(0, "#width: "+str(width) + " depth:" +str(depth))
    script.insert(0, "#vec: "+str(width*depth))
    script.insert(0, "#total: "+str(width*depth+num_input+width))
    return script
    
with open("net_"+str(depth)+"-"+str(width)+".daph", "w") as f:
    for line in generate_script():
        f.write(line + '\n')