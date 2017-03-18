#coding=utf8
import numpy as np
from predict import *
import copy

def getSize(i ,j):
    return i * 0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidGradient(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

def pack_thetas(t1, t2):
    # return np.concatenate((t1.reshape(-1), t2.reshape(-1)))
    return np.concatenate( (t1.flatten(), t2.flatten()), axis=1 ).T

def nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, l):
    m = X.shape[0]  #训练集大小
    J = 0 
    Theta1_grad = np.fromfunction(getSize, (Theta1.shape[0], Theta1.shape[1]))
    Theta2_grad = np.fromfunction(getSize, (Theta2.shape[0], Theta2.shape[1]))

    Y = np.zeros([m, num_labels])

    for i in range(m):
        Y[i, y[i]-1] = 1 #Y[0]代表1， Y[1]代表2， ... Y[9]代表0

    (p, h1, h2) = predict(Theta1, Theta2, X, False)

    tmp = np.dot(np.log(h2), -Y.T) - np.dot(np.log(1-h2), (1-Y.T))
    tmp = np.multiply(tmp, np.eye(m))

    sumJ = 0
    for i in range(m):
        sumJ += tmp[i, i]

    T1 = copy.deepcopy(Theta1) #注意要用深拷贝
    # T1 = Theta1
    T1[:, 0] = 0
    T2 = copy.deepcopy(Theta2)
    # T2 = Theta2
    T2[:, 0] = 0
    J = 1.0 * sumJ / m + 1.0*l/(2*m) * ( np.sum(pow(T1, 2)) + 
        np.sum(pow(T2, 2)) )

    Delta_2 = np.zeros([num_labels, hidden_layer_size+1])
    Delta_1 = np.zeros([hidden_layer_size, input_layer_size+1])

    biasUnit = np.ones([m, 1])
    t1 = np.dot(np.concatenate((biasUnit, X), axis=1), Theta1.T)
    sG = sigmoidGradient(np.concatenate((biasUnit, t1), axis=1))
    for i in range(m):
        example = np.matrix(X[i])
        (p, h1, h2) = predict(Theta1, Theta2, example, False)
        sG_t = np.matrix(sG[i]).T
        delta3 = h2.T - np.matrix(Y[i, :]).T
        delta2 = np.multiply( np.dot(Theta2.T, delta3), sG_t )
        row = delta2.shape[0]
        delta2 = delta2[1:row, :]

        biasUnit = np.ones([example.shape[0], 1])
        Delta_2 = Delta_2 + np.dot(delta3, np.concatenate((biasUnit, h1), axis=1)) 
        Delta_1 = Delta_1 + np.dot(delta2, np.concatenate((biasUnit, example), axis=1)) 
    
    Theta1_grad = Delta_1 * 1.0/m + 1.0*l/m * T1
    Theta2_grad = Delta_2 * 1.0/m + 1.0*l/m * T2

    # grad = pack_thetas(Theta1_grad, Theta2_grad)
    print('Cost = ' + str(J))

    return J, np.array(Theta1_grad), np.array(Theta2_grad)
