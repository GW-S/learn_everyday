import torch
device = torch.device('cpu')
N,D_in,H,D_out = 64,1000,100,10
#N:     迭代次数
#D_in:  输入的纬度
#D_out: 输出的纬度
#H隐藏层
x = torch.randn(N,D_in, device=device)  # 返回一个tensor
y = torch.randn(N,D_out,device=device)  # H 是隐藏层
w1 = torch.randn(D_in,H, device=device,requires_grad =True)
w2 = torch.randn(H,D_out,device=device,requires_grad =True) # H 是隐藏层
learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)            # 乘法运算
    h_relu = h.clamp(min=0.1) # 压缩向量到0
    print(h_relu)
    y_pred = h_relu.mm(w2)
    loss =(y_pred-y).pow(2).sum()
    print(t,loss.item()) # 就算是一个标量，也存储在item中
    loss.backward()
    with torch.no_grad():
        w1 -=learning_rate * w1.grad
        w2 -=learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
# https://blog.csdn.net/weixin_38791178/article/details/80610629

#官方文档
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
