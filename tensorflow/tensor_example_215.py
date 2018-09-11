# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/9
"""
深度学习入门 实例55
"""

import numpy as np
import copy
np.random.seed(0)

def sigmoid(x):
    output = 1/(1 + np.exp(-x))                   # todo(): 知识点：sigmoid的x取值为负无穷到正无穷
    return output                                 # todo():        sigmoid的范围是0-1

def sigmoid_output_to_derivative(output):
    return output * (1-output)

int2binary = {}

binary_dim = 8
largest_number = pow(2, binary_dim)
binary = np.unpackbits (np.array([range(largest_number)], dtype=np.uint8).T, axis= 1)
                                                                # unpackbits ，将正常的解包
                                                                # todo() 为什么转置成这个样子？

print(np.array([range(largest_number)], dtype=np.uint8).T)
print(binary)

for i in range(largest_number):
    int2binary[i] = binary[i]

# 于是这就很容易腿断了 【 【1，2，3，4，5，6，7，8】，【】 】


# 参数设置
alpha = 0.9
input_dim  = 2
hidden_dim = 16
output_dim = 1

# 初始化网络
synapse_0 = (2 * np.random.random((input_dim, hidden_dim)) - 1)  * 0.05          # shape(input_dim,hidden_dim)
synapse_1 = (2 * np.random.random((hidden_dim,output_dim)) - 1) * 0.05           # shape(input_dim,output_dim)
synapse_h = (2 * np.random.random((hidden_dim,output_dim)) - 1) * 0.05           # shape(hidden_dim,output_dim)

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)


# 实际值，所有值都在这个字典里，只是取个值罢了
for j in range(10000):
    a_int = np.random.randint(largest_number)       # 生成一个数字a
    b_int = np.random.randint(largest_number/2)     # 生成一个数字b
    if a_int < b_int:
        tt = a_int
        a_int = b_int
        b_int = tt
    a = int2binary[a_int]
    b = int2binary[b_int]
    c_int = a_int - b_int
    c = int2binary[c_int]

#
d = np.zeros_like(c)
overallError = 0

layer_2_deltas = list()         # delta 变量，增量       存储每个时间点输出层的误差
layer_1_values = list()         # 存储每个时间点隐藏层的值

layer_1_values.append(np.ones(hidden_dim) * 0.1)   # 对于第一个，由于不知道隐藏值，所以进行初始化


# 正向传播
for position in range(binary_dim):          # 对于每一个二进制位
    X = np.array( [ [a[binary_dim - position - 1],b[binary_dim - position -1]] ])    # 实质上是 [1,2]
    # 从右到左，每次取两个输入数字的一个bit位
    y = np.array([[c[binary_dim - position -1]]]).T   # 为什么要转置？是怎么转的？  y 原本是[3],（1，） 经过转置,就是【【3】】（1，1），这就变成二维的了
    #（X，synapse_0) 就是一个synapse_0 ,
    layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))       # hidden层 到 input 和 hidden
    layer_2 = sigmoid(np.dot(layer_1,synapse_1))   #（0，1）                                       # 15 个

    layer_2_error = y - layer_2
    # 预测误差
    layer_2_deltas.append( (layer_2_error) * sigmoid_output_to_derivative(layer_2) )

    overallError += np.abs(layer_2_error[0])        # synapse 突触，神经元

    d[binary_dim - position - 1] = np.round(layer_2[0][0])

    # 将隐藏层保存起来。
    layer_1_values.append(copy.deepcopy(layer_1))

future_layer_1_delta = np.zeros(hidden_dim)




# # 初始化为0
#
# # 反向传播
for position in range(binary_dim):

    X = np.array([ [a[position],b[position]] ] )#
    layer_1 = layer_1_values[-position-1]       # 当前时间点的隐藏层
    prev_layer_1 = layer_1_values[-position-2]  # 前一个时间点的隐藏层

    layer_2_delta = layer_2_deltas[-position-1]    # 输出层导数，计算隐藏层误差


    # 计算隐藏层的误差
    layer_1_delta = (future_layer_1_delta.reshape(16,1).dot(synapse_h.T) + layer_2_deltas.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
    # 误差的和，除以隐藏层误差

    # 暂存更新矩阵存起来
    synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
    synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
    synapse_0_update += X.T.dot(layer_1_delta)

    future_layer_1_delta = layer_1_delta

# 完成所有反向传播之后
synapse_0 += synapse_0_update * alpha
synapse_1 += synapse_1_update * alpha
synapse_h += synapse_h_update * alpha

synapse_0_update *= 0
synapse_1_update *= 0
synapse_h_update *= 0

if(j%800 ==0):
    print("总误差：" + str(overallError))
    print("pred:" + str(d))
    print("pred:" + str(c))

    out = 0

    for index,x in enumerate(reversed(d)):
        out += x*pow(2,index)
    print(str(a_int) + " - " + str(b_int) + "=" + str(out))

    print("-----------")
















