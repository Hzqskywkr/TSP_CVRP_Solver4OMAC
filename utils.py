import numpy as np

def Q_Matrix_1(n):
    Q = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            Q[i * n + j, i * n + j] += -1
            for k in range(j + 1, n):
                Q[i * n + j, i * n + k] += 2

    for i in range(n):
        for j in range(n):
            Q[j*n+i,j*n+i] += -1
            for k in range(j+1,n):
                Q[j * n + i, k * n + i] += 2

    Q = (Q+Q.T)/2

    return Q

def Q_Matrix_2(n, Q, edge_null):
    for edge in edge_null:
        a = edge[0]
        b = edge[1]
        for k in range(n):
            if k == n-1:
                Q[a * n + k, b * n] += 1
                Q[b * n + k, a * n] += 1
            else:
                Q[a*n+k,b*n+k+1] += 1
                Q[b * n + k, a * n + k + 1] += 1
    Q = (Q + Q.T) / 2

    return Q

def Q_Matrix_3(n, Q, edges, weight):
    for edge in edges:
        a = edge[0]
        b = edge[1]
        for k in range(n):
            if k == n-1:
                Q[a * n + k, b * n] += 1*weight[a,b]
                Q[b * n + k, a * n] += 1 * weight[b, a]
            else:
                Q[a*n+k,b*n+k+1] += 1*weight[a,b]
                Q[b * n + k, a * n + k + 1] += 1 * weight[b, a]
    Q = (Q + Q.T) / 2

    return Q


def Q_Matrix_2_I(n, Q, edge_null):
    for edge in edge_null:
        a = edge[0]
        b = edge[1]
        for k in range(n):
            Q[a * n + k, b * n + k + 1] += 1
            Q[b * n + k, a * n + k + 1] += 1
    Q = (Q + Q.T) / 2

    return Q


def Q_Matrix_2_II(n, Q, edge_null_0):
    for edge in edge_null_0:
        b = edge[1]
        Q[n * (b - 1), n * (b - 1)] += 1
        Q[n * (b - 1) + n - 1, n * (b - 1) + n - 1] += 1
    Q = (Q + Q.T) / 2

    return Q


def Q_Matrix_3_I(n, Q, edges, weight):
    for edge in edges:
        a = edge[0]
        b = edge[1]
        for k in range(n - 1):
            Q[a * n + k, b * n + k + 1] += 1 * weight[a, b]
            Q[b * n + k, a * n + k + 1] += 1 * weight[b, a]
    Q = (Q + Q.T) / 2

    return Q


def Q_Matrix_3_II(n, Q, edges_0, weight0):
    for edge in edges_0:
        b = edge[1]
        Q[n * (b - 1), n * (b - 1)] += 1 * weight0[b - 1]
        Q[n * (b - 1) + n - 1, n * (b - 1) + n - 1] += 1 * weight0[b - 1]
    Q = (Q + Q.T) / 2

    return Q