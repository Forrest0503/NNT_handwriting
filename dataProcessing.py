#coding=utf8
import os
from PIL import Image
import scipy.io as sio  
import numpy as np

def initialDataset(cv):
    if not cv:
        baseDir = '/Users/Jason/Developer/ML/handwriting/NNT/'
        #初始化X和y
        I_init = Image.open('/Users/Jason/Developer/ML/handwriting/NNT/1/1_3.JPG')
        im_vec = imgToVec(I_init)
        X = im_vec
        y = np.matrix('1')
        for i in range(1, 11):
            (X, y) = addDataByFolder(baseDir, i, X, y)
        return X, y
    else:
        baseDir = '/Users/Jason/Developer/ML/handwriting/NNT_CV/'
        #初始化X和y
        I_init = Image.open('/Users/Jason/Developer/ML/handwriting/NNT_CV/1/1_1.JPG')
        im_vec = imgToVec(I_init)
        Xval = im_vec
        yval = np.matrix('1')
        for i in range(1, 11):
            (Xval, yval) = addDataByFolder(baseDir, i, Xval, yval)
        return Xval, yval

def addDataByFolder(baseDir, suffix, X, y):
    folder = baseDir + str(suffix)
    assert os.path.exists(folder)
    assert os.path.isdir(folder)
    imageList = os.listdir(folder)
    imageList = [os.path.join(folder, item) for item in imageList if os.path.isfile(os.path.join(folder, item)) and ('.JPG' in item  or '.jpg' in item)]
    
    for each in imageList:
        img = Image.open(each)
        img_vec = imgToVec(img)
        for i in range(1):
            X = np.concatenate((X, img_vec), axis=0)
            y = np.vstack((y, np.matrix(str(suffix))))
    return X, y

def imgToVec(img):
    L = img.convert('L')
    newImg = L.resize((20, 20))
    # newImg.show()
    im_array = np.array(newImg)
    # 将样本归一化
    im_vec = np.matrix( 1.0*(im_array - (1.0*255/2)) / (1.0*255/2) ).reshape(-1)
    #二值化消除噪音
    arr = im_vec.getA()
    for i in range(400):
        if arr[0][i] < 0:
            arr[0][i] = -1
        else:
            arr[0][i] = 1
    im_vec = im_vec * -1
    return im_vec

def vecToImg(vec):
    img_matrix = vec.reshape((20, 20))
    I = Image.fromarray(img_matrix*(1.0*255/2) + (1.0*255/2))
    I.show()