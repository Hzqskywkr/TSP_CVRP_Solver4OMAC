import numpy as np
import matplotlib.pyplot as plt
import time
from Read import read_CVRP
from TSP_Solver import TSP_Solver
import math
import copy

CVRP_files = ['P-n19-k2.txt', 'E-n33-k4.txt','E-n51-k5.txt']
K = 5
n,Q, NODE_COORD,DEMAND = read_CVRP(CVRP_files[2])
print('n',n)
print('Q',Q)
DEMAND = DEMAND[1:]
X_depot = NODE_COORD[0,0]
Y_depot = NODE_COORD[0, 1]
custmer_list = np.zeros((n-1,3))
custmer_list[:,0:-1] = NODE_COORD[1:]
custmer_list[:,2] = DEMAND
D = np.zeros(n-1)
def distance(CC,custmer):
    temp = (CC[0]-custmer[0])**2+(CC[1]-custmer[1])**2
    dis = math.sqrt(temp)
    return dis
custmer_list = custmer_list[custmer_list[:,2].argsort()[::-1]]

def findmax(custmer_list):
    demand = custmer_list[:,2]
    Qm = max(demand)
    index_Qm_list = []
    for i in range(len(demand)):
        if demand[i] == Qm:
            index_Qm_list.append(i)
    index_Qm = np.random.choice(index_Qm_list)
    return index_Qm

def center(custer_lists):
    l = len(custer_lists)
    CC = [0,0,0]
    for custer in custer_lists:
        CC[0] += custer[0]
        CC[1] += custer[1]
        CC[2] += custer[2]
    CC[0] = CC[0]/l
    CC[1] = CC[1] / l
    return CC

def mer_center(CC,custmer,l):
    CC[0] = (CC[0]*(l-1)+custmer[0])/l
    CC[1] = (CC[1]*(l-1)+custmer[1])/l
    CC[2] = CC[2] + custmer[2]
    return CC

def K_means(custmer_list):
    Inicustms_index = np.random.choice(n-1, K,replace=False)
    Custers_list = []
    for i in range(K):
        Custers_list.append([])
    IniCC = []
    for i in range(K):
        ith_custer = []
        ith_custer.append(custmer_list[Inicustms_index[i]].tolist())
        #Custers_list.append(ith_custer)
        IniCC.append(center(ith_custer))
    for C in IniCC:
        C[2] = 0
    CC = IniCC
    CC_old = copy.deepcopy(CC)
    niter = 15
    for i in range(niter):
        CC, Custers_list = Custer(CC,custmer_list)
        DCC = np.array(CC)-np.array(CC_old)
        #print('i,DCC', i, DCC)
        E = DCC[:,0].T@DCC[:,0]+DCC[:,1].T@DCC[:,1]
        for C in CC:
            C[2] = 0
        CC_old = copy.deepcopy(CC)
    return CC, Custers_list

def Custer(CC,custmer_list):
    Custers_list = []
    for i in range(K):
        Custers_list.append([])
    for custmer in custmer_list:
        dis = np.zeros((K,2))
        for i in range(K):
            dis[i,0] = i
            dis[i,1] = distance(custmer,CC[i])
        dis = dis[dis[:,1].argsort()]
        #print('dis', dis)
        for i in range(K):
            dem = custmer[2]
            nearst_CC = int(dis[i,0])
            if CC[nearst_CC][2]+dem<=Q:
                Custers_list[nearst_CC].append(custmer.tolist())
                CC[nearst_CC][2] += dem
                break
    for i in range(K):
        CC[i] = center(Custers_list[i])

    return CC, Custers_list


def Gen_TSP_weight(TSP_weight):
    n = len(TSP_weight)
    weight = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            weight[i,j] = distance(TSP_weight[i],TSP_weight[j])

    return weight

def main():
    startTime = time.perf_counter()
    CC, Custers_list = K_means(custmer_list)
    endTime = time.perf_counter()
    time_K_means = endTime - startTime
    print('time for K_means',time_K_means)
    CC_list = []

    plt.scatter(X_depot, Y_depot, c='black')
    cost = 0
    count = 0
    check_custmer = []
    for custer in Custers_list:
        print('custer',custer)
        count += len(custer)
        for custmer in custer:
            check_custmer.append(custmer)
        n = len(custer)+1
        TSP_COORD = [[X_depot,Y_depot]]
        for TSP_custer in custer:
            temp = TSP_custer[0:2]
            TSP_COORD.append(temp)
        TSP_weight = Gen_TSP_weight(TSP_COORD)
        eng, hard, rout = TSP_Solver(n,TSP_weight)
        if hard == 0:
            cost += eng
            print('rout is', rout)
        else:
            print('hard constraints not sat ')
        CC = center(custer)
        CC_list.append(CC)
        X = [i[0] for i in custer]
        Y = [i[1] for i in custer]
        XX = [X_depot]
        YY = [Y_depot]
        for x in X:
            XX.append(x)
        for y in Y:
            YY.append(y)
        Xrout = []
        Yrout = []
        rout = rout-1
        for pos in rout:
            Xrout.append(XX[pos])
            Yrout.append(YY[pos])
        Xrout.append(XX[rout[0]])
        Yrout.append(YY[rout[0]])
        plt.scatter(X, Y)
        plt.plot(Xrout, Yrout, '--')
    print('total cost is', cost)
    print('count',count)
    plt.show()
    print(CC_list)
    print('check_custmer',check_custmer)
    for custmer in custmer_list:
        if custmer.tolist() in check_custmer:
            continue
        else:
            print('notin',custmer)
    print("Done")





if __name__ == "__main__":
    main()
