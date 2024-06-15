import numpy as np
from omacMatmul.matmul import oMAC_matmul
import time
from utils import Q_Matrix_1
from utils import Q_Matrix_2
from utils import Q_Matrix_3

def quad_K(K):
    Bit = 4
    SL = -2**(Bit-1)
    SR = 2**(Bit-1)-1
    KL = K.min()
    KR = K.max()
    epsilon = 1E-6
    if KR<0:
        scale = SL/KL
    elif KL>0:
        scale = SR/KR
    else:
        scale = min(SL/(KL+epsilon),SR/(KR+epsilon))
    KS = np.around(K*scale)

    return scale, KS

def getKL(Q,MATRIX_SIZE):
    # This function get the matrix K and external field L
    # Calculate K matrix for Q Matrix
    K = Q.copy()
    K = -0.5 * K
    for i in range(MATRIX_SIZE):
        K[i][i] = 0
    L = np.zeros((MATRIX_SIZE, 1))  # external magnetic field
    L = np.sum(Q, axis=1)
    L = -0.5 * L

    return K, L

def SvectorInitialization(S):
    for i in range(S.shape[0]):
        val = np.random.randint(0, 2 ** 15) % 2
        S[i] = val

    return S

def calculateEnergy(S, S_PIC, L, Qmax,Const):
    energy = 0
    for i in range(S.shape[0]):
        if S[i] == 0:
            energy += S_PIC[i]
    for i in range(S.shape[0]):
        if S[i] == 1:
            energy -= L[i]

    energy = 2 * energy+Const
    energy = energy*Qmax
    #energy = int(energy+0.5)  #四舍五入取整

    return energy

def getThresholds(K, L):
    matrix = np.zeros((K.shape[0]))
    for i in range(K.shape[0]):
        matrix[i] = K[i].sum() - L[i]
    #matrix = np.round(matrix)
    #matrix = matrix.astype(np.int)
    return matrix

def isVectorZero(S):
    return np.all(S == 0)

def isVectorOne(S):
    return np.all(S == 1)

def compareToThresholds(S, thresholds):
    n = S.shape[0]
    for i in range(n):
        if S[i] > thresholds[i]:
            S[i] = 1
        else:
            S[i] = 0

def findminindex(DH):
    minDH = np.min(DH)
    index_minSigma = []
    for i in range(DH.shape[0]):
        if DH[i] == minDH:
            index_minSigma.append(i)
    return index_minSigma

def Flip(S,S_PIC, thresholds2):
    Sigma = 2 * S - 1
    DM = 2 * S_PIC.T - thresholds2
    DH = 2 * Sigma.T * DM  # (-2)*Sigma'*DM
    minSigma = findminindex(DH)
    index = np.random.choice(minSigma)
    Sigma[index] = -Sigma[index]
    S = (Sigma + 1) / 2
    return S

def check(n, MATRIX_SIZE, S):
    rout = np.zeros(n, dtype=np.int)
    hard = 0
    ind_check = []
    for i in range(0, MATRIX_SIZE, n):
        x = S.T[i:i + n]
        none = np.sum(x == 1)
        if none == 1:
            ind = np.where(x == 1)[0][0]
            if ind in ind_check:
                hard = 1
                return hard, rout
            else:
                ind_check.append(ind)
                rout[ind] = int(i / n)
        else:
            hard = 1
            return hard, rout
    return hard, rout

def Distance(n, rout, weight):
    distance = 0
    for i in range(n):
        if i == n-1:
            distance += weight[rout[i], rout[0]]
        else:
            distance += weight[rout[i],rout[i+1]]
    return distance

def runIsingAlgorithm_simulator(niter, n, MATRIX_SIZE, weight, K, L, KS, scale, Scale_S):
    # Initialization stats S and energy
    S = np.zeros((256), dtype=np.int)
    SvectorInitialization(S)
    # Calculate initial energy
    best_matrix = S
    best_energy = 1E9
    # Using the adjacency matrix to set the thresholds
    thresholds = getThresholds(K, L)
    KS_Pace = np.pad(KS,((0,256-MATRIX_SIZE),(0,256-MATRIX_SIZE)),'constant',constant_values = 0)
    KS_Pace = KS_Pace.T.reshape(1,256,256)
    t_MVM = 0
    omatmul = oMAC_matmul()
    omatmul.init()
    for i in range(niter):
        ##Calculate matrix multiplication and energy
        '''
        S_PIC = Ks @ (S*Scale_S)
        for j in range(Ks.shape[0]):
            noise = np.random.randn() * g
            S_PIC[j] = S_PIC[j] + noise
        '''
        S_Pace = Scale_S*S.reshape(1,1,256)
        startTime_MVM = time.perf_counter()
        S_PIC, latency = omatmul(S_Pace, KS_Pace)
        endTime_MVM = time.perf_counter()
        t_MVM += endTime_MVM - startTime_MVM
        S_PIC = S_PIC.reshape(256,)
        S_PIC = S_PIC / (scale*Scale_S)
        hard, rout = check(n, MATRIX_SIZE, S[0:MATRIX_SIZE])
        if not hard:
            energy = Distance(n, rout, weight)
            if energy < best_energy and not isVectorZero(S) and not isVectorOne(S):
                best_matrix = S.copy()
                best_energy = energy
        # Updata the state S
        S[0:MATRIX_SIZE] = Flip(S[0:MATRIX_SIZE], S_PIC[0:MATRIX_SIZE], thresholds)
        S = S.astype(int)

    return best_matrix, best_energy

def TSP_Solver(n,TSP_weight):
    A = np.max(TSP_weight) * n
    QA = Q_Matrix_1(n)
    edge_null = []
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if TSP_weight[i, j] == 0:
                edge_null.append([i, j])
            else:
                edges.append([i, j])
    QA = Q_Matrix_2(n, QA, edge_null)
    QA = A * QA
    QB = np.zeros((n * n, n * n))
    QB = Q_Matrix_3(n, QB, edges, TSP_weight)
    Q = QA + QB
    Qmax = abs(Q).max()
    Q = Q / Qmax
    NITER = 20000
    MATRIX_SIZE = n * n
    K, L = getKL(Q,MATRIX_SIZE)
    scale, Ks = quad_K(K)
    Scale_S = 8
    best_matrix, best_energy = runIsingAlgorithm_simulator(NITER, n, MATRIX_SIZE, TSP_weight, K, L, Ks, scale, Scale_S)
    rout = np.zeros(n, dtype=np.int)
    hard = 0
    for i in range(0, MATRIX_SIZE, n):
        x = best_matrix.T[i:i + n]
        none = np.sum(x == 1)
        if none == 1:
            ind = np.where(x == 1)[0][0]
            rout[ind] = int(i / n) + 1
        else:
            hard = 1
            break

    return best_energy, hard, rout
