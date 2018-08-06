import numpy as np

# recommand movies with top 5 estimate ratings
def recommandation_by_MF_bias(matrix, userid):
    P, Q, mean_value, bu, bi = matrix_factorization_bias(matrix)
    ematrix = np.matmul(P, Q)
    for i in range(len(ematrix)):
        for j in range(len(ematrix[i])):
            ematrix[i][j] += bu[i] + bi[j] + mean_value
    L = ematrix[userid - 1]
    orginal = np.asarray(matrix.iloc[userid-1])
    recommed = {}
    for i in range(len(orginal)):
        if orginal[i] != 0:
            recommed[i+1] = L[i]
    recommed = sorted(recommed.items(), key=lambda item:item[1], reverse=True)
    return recommed[:5]

# recommand movies with top 5 estimate ratings by basic matrix factorization
def recommandation_by_MF(matrix, userid):
    P, Q = matrix_factorization(matrix)
    ematrix = np.matmul(P, Q)
    L = ematrix[userid - 1]
    orginal = np.asarray(matrix.iloc[userid-1])
    recommed = {}
    for i in range(len(orginal)):
        if orginal[i] != 0:
            recommed[i+1] = L[i]
    recommed = sorted(recommed.items(), key=lambda item:item[1], reverse=True)
    return recommed[:5]

# matrix factorization with bias
def matrix_factorization_bias(R, steps=100, alpha=0.01, beta=0.1):
    new_matrix = np.asarray(R)
    np.random.seed(11)
    P = np.random.rand(len(new_matrix), 100)
    Q = np.random.rand(len(new_matrix[0]), 100).T
    bu = np.zeros(len(new_matrix)).astype(np.int64)
    bi = np.zeros(len(new_matrix[0])).astype(np.int64)
    L = []
    num = 0
    for i in range(len(new_matrix)):
        for j in range(len(new_matrix[i])):
            if new_matrix[i][j] != 0:
                num += new_matrix[i][j]
                L.append((i, j, new_matrix[i][j]))
    mean_value = num / len(L)
    for step in range(steps):
        print("step: {}/{}".format(step,steps))
        np.random.shuffle(L)
        for x in L:
            X = P[x[0],:]
            Y = Q[:,x[1]]
            e = x[2] - np.clip(np.matmul(X,Y)+bu[x[0]]+bi[x[1]]+mean_value,0,5)
            bu[x[0]] = bu[x[0]] + alpha * (e - beta * bu[x[0]])
            bi[x[1]] = bi[x[1]] + alpha * (e - beta * bi[x[1]])
            temp_X = X + alpha * (e * Y.T - beta * X)
            temp_Y = Y + alpha * (e * X.T - beta * Y)
            P[x[0], :] = temp_X
            Q[:, x[1]] = temp_Y
    return P, Q, mean_value, bu, bi

# basic matrix factorization
def matrix_factorization(R, steps=100, alpha=0.01, beta=0.1):
    new_matrix = np.asarray(R)
    np.random.seed(11)
    P = np.random.rand(len(new_matrix), 100)
    Q = np.random.rand(len(new_matrix[0]), 100).T
    L = []
    num = 0
    for i in range(len(new_matrix)):
        for j in range(len(new_matrix[i])):
            if new_matrix[i][j] != 0:
                num += new_matrix[i][j]
                L.append((i, j, new_matrix[i][j]))
    for step in range(steps):
        print("step: {}/{}".format(step,steps))
        np.random.shuffle(L)
        for x in L:
            X = P[x[0],:]
            Y = Q[:,x[1]]
            e = x[2] - np.clip(np.matmul(X,Y),0,5)
            temp_X = X + alpha * (e * Y.T - beta * X)
            temp_Y = Y + alpha * (e * X.T - beta * Y)
            P[x[0], :] = temp_X
            Q[:, x[1]] = temp_Y
    return P, Q

# estimate by matrix factorization with bias
def estimate_MF_bias(test_data, matrix):
    P, Q, mean_value, bu, bi = matrix_factorization_bias(matrix)
    ematrix = np.matmul(P, Q)
    for i in range(len(ematrix)):
        for j in range(len(ematrix[i])):
            ematrix[i][j] += bu[i] + bi[j] + mean_value
    estimate_data = {}
    for x in test_data.keys():
        estimate_data[x] = ematrix[x[0]-1][x[1]-1]
    return estimate_data

# estimate by basic matrix factorization
def estimate_MF(test_data, matrix):
    P, Q = matrix_factorization(matrix)
    ematrix = np.matmul(P, Q)
    estimate_data = {}
    for x in test_data.keys():
        estimate_data[x] = ematrix[x[0] - 1][x[1] - 1]
    return estimate_data
