# 在Reshaping Tensors

# cat();
# view();


# torch 看到这里
# https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html



# wordvector 能做什么？ 能features and words

# combine these representations?


# wordvect 相似的会

# dense outputs from our neural network

# 一个稀疏的数据，我们要把它变为密集的，这里面的代价是怎么样的？
# wordvector 能够将字符压缩成一个部分，相近的东西距离相似

# 一个原则，在一个位置上相似的，与其他位置上的东西

# 那么怎么做呢？

# 人们用

# thousands

# 聚类其实是没有办法的办法

# 有没有别人训练好的embedding 模型？？


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


word_to_ix = {"hello":0,"world":1}

embeds = nn.Embedding(2,5)

