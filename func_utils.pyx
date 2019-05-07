# coding:utf-8
#import math

def get_cosine_value(X_list, Y_list, X_norm, Y_norm):

    # 分子 x1*y1 + x2*y2 + ... + xn*yn
    # 分母 X_norm * Y_norm

    #print(X_list)
    #print(Y_list)

    if (X_norm <= 0.0 or Y_norm <= 0.0 or len(X_list) != len(Y_list)):
        return 0

    X = X_list.reshape(1, 256)
    Y = Y_list.reshape(1, 256)

    return float(X.dot(Y.T) / (X_norm * Y_norm))
