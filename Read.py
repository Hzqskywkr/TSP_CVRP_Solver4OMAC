import numpy as np
import re
import string

def read_CVRP(CVRP_file):
    with open(CVRP_file) as f:
        print(f"[READ] {CVRP_file}")
        n = 0
        Q = 0
        count = 0
        for line in f.readlines():
            count += 1
            line = line.strip(' \n')
            line_cont=line.split()
            if line_cont[0]=='DIMENSION':
                n = int(line_cont[2])
                NODE_COORD = np.zeros((n,2))
                DEMAND = np.zeros(n)
            if line_cont[0]=='CAPACITY':
                Q = int(line_cont[2])
            if count>=8 and count<=7+n:
                #print('line_cont',line_cont)
                NODE_COORD[count - 8, 0] = float(line_cont[1])
                NODE_COORD[count - 8, 1] = float(line_cont[2])
            if count>=9+n and count<=8+2*n:
                #print('line_cont', line_cont)
                DEMAND[count-9-n] = float(line_cont[1])

    return n, Q, NODE_COORD, DEMAND