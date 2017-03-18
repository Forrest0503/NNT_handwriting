#coding=utf8
import numpy as np
import math

def fill(i ,j):
    return i * 0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict(Theta1, Theta2, X, flag):
    #初始化
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.fromfunction(fill, (X.shape[0], 1))
    # p = np.matrix(p)

    biasUnit = np.ones([X.shape[0], 1])

    h1 = sigmoid(np.dot(np.concatenate((biasUnit, X), axis=1), Theta1.T)) #隐含层的输出
    h2 = sigmoid(np.dot(np.concatenate((biasUnit, h1), axis=1), Theta2.T)) #输出层的输出
    if flag == True:
        for i in range(m):
            arr = h2[i].tolist()
            maxP = max(arr[0])
            # p[i] = h2[i].argmax(axis=0)
            p[i] = arr[0].index(maxP)

    return (p, h1, h2)