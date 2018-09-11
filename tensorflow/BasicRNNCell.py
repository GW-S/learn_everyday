# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/4

import tensorflow as tf

import copy, numpy as np
np.random.seed(0)

def sigmoid(x):
    output = 1/(1 + np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)

int2binary = {}
binary_dim = 8
largest_number = pow(2,binary_dim)

binary = np.unpackbits()



