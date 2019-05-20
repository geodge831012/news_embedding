# coding:utf-8
import numpy as np

#########################################################################
## sigmoid函数 ##
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## sigmoid函数 ##
def matrix_dot(matrix, x):
    return matrix.dot(x)
