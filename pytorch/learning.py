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

lookup_tensor = torch.tensor([word_to_ix["hello"]],dtype= torch.long)

hello_embed = embeds(lookup_tensor)

print(hello_embed)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [([test_sentence[i],test_sentence[i+1]] , test_sentence[i+2]) for i in range(len(test_sentence)-2)]
print(trigrams[:3])

vocab = set(test_sentence)


