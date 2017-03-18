#coding=utf8
import numpy as np
import scipy.io as sio  
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
from PIL import Image
from scipy.optimize import fmin_bfgs
from predict import *
from dataProcessing import *
from nnCostFunction import *
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    def __init__(self):
        #神经网络参数
        self.input_layer_size  = 400  # 20x20 Input Images of Digits
        self.hidden_layer_size = 25   # 25 hidden units
        self.num_labels = 10          # 10 labels, from 1 to 10 
        self.norm_lambda = 1          # normalization parameter

        (self.X, self.y) = initialDataset(cv=False) #初始化训练集
        (self.Xval, self.yval) = initialDataset(cv=True) #初始化验证集
        self.errors = [] #训练集误差
        self.errors_CV = [] #验证集误差

    def loadTheta(self):
        t1 = np.load('Theta1.npy')
        t2 = np.load('Theta2.npy')
        return (t1, t2)

    def startTraining(self):
        maxIter = 30000  #最大迭代次数
        learningRate = 0.3  #学习速率
        lastJ = 100000 

        #随机初始化权重
        t1 = self.__randomInitializeWeights(self.hidden_layer_size, self.input_layer_size+1) 
        t2 = self.__randomInitializeWeights(self.num_labels, self.hidden_layer_size+1)

        for i in range(maxIter):
            (J, t1, t2) = self.__gradientDescent(t1, t2, learningRate)
            if J < 0.6:
                print('training finished!')
                break
            if abs(lastJ - J) < 0.0001:
                print('Converged!')
                break
            lastJ = J
            if i%10 == 0 and i != 0:
                self.errors.append( self.testError(self.X, self.y, t1, t2, cv=False) )
                self.errors_CV.append( self.testError(self.Xval, self.yval, t1, t2, cv=True) )
            
        np.save('Theta1.npy', t1)
        np.save('Theta2.npy', t2)
        self.plotError(index=(i/10))
        return t1, t2

    def startPredicting(self, theta1, theta2):
        while True:
            path = raw_input('请拖入要识别的图片')
            I = Image.open(path[0:len(path)-1])
            self.predictOnNewImg(theta1, theta2, imgToVec(I))

    def predictOnNewImg(self, t1, t2, img_vec):
        result = predict(t1, t2, img_vec, True)
        (p, _, _) = predict(t1, t2, np.matrix(img_vec), True)
        print( int(p[0][0]+1)%10 )

    def __randomInitializeWeights(self, row_num, col_num):
        epsilon = 1.0*math.sqrt(6)/math.sqrt(row_num+col_num);
        W = np.random.random((row_num, col_num)) * 2 * epsilon - epsilon
        return W

    def __gradientDescent(self, t1, t2, alpha):
        (J, grad1, grad2) = nnCostFunction(t1, t2, self.input_layer_size, self.hidden_layer_size, self.num_labels, self.X, self.y, self.norm_lambda)

        t1 = t1 - alpha * grad1
        t2 = t2 - alpha * grad2

        return (J, t1, t2)

    def testError(self, X, y, t1, t2, cv):
        total = X.shape[0]  #采样数量
        acc = 0
        for i in range(total):
            test = X[i]
            (p, _, _) = predict(t1, t2, np.matrix(test), True)
            if int(p[0][0]+1)%10 == np.array(y)[i][0]%10:
                acc += 1
        if not cv:
            print('Training Set Error: ' + str(1 - 1.0*acc/total))
        else:
            print('Cross validation Set Error: ' + str(1 - 1.0*acc/total))
        return 1 - 1.0*acc/total

    def plotError(self, index):
        plt.plot(range(index), self.errors, label='Training Set')
        plt.plot(range(index), self.errors_CV, 'red', label='Cross Validation Set')
        plt.xlabel('iteration')
        plt.ylabel('error')
        plt.legend()
        plt.show()



nnNetwork = NeuralNetwork()
(t1, t2) = nnNetwork.startTraining()
# (t1, t2) = nnNetwork.loadTheta()
nnNetwork.startPredicting(t1, t2)
