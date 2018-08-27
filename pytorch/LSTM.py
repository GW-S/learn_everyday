import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3,3)

inputs = [torch.randn(1,3) for _ in range(5)]       # 这种写法有点意思

print(inputs)

hidden = (torch.randn(1,1,3),torch.randn(1,1,3))

print(hidden)

for i in inputs:
    out,hidden = lstm(i.view(1,1,-1),hidden)

    # i =( [[1,2,3]]
    # i.view(1,1,-1) = [[[1,2,3]]] 这是

    #print(i.view(1,1,-1))

inputs = torch.cat(inputs).view(len(inputs),1,-1)

hidden = (torch.randn(1,1,3),torch.randn(1,1,3))
out,hidden = lstm(inputs,hidden)

print(out)
print(hidden)





