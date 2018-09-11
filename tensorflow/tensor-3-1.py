# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/6


# 实现线性回归

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

train_X = np.linspace(-1,1,100)  # space 把……分隔开

train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

plt.plot(train_X,train_Y,'ro',label='Original data')

plt.legend()

plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]),name = "weight")
b = tf.Variable(tf.zeros([1]),name= "bias")

z = tf.multiply(X,W) + b

tf.summary.histogram('z',z)

cost = tf.reduce_mean(tf.square(Y-z))   # 牛逼

tf.summary.scalar('loss_function',cost)


learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # 梯度下降器


saver = tf.train.Saver()

init = tf.global_variables_initializer()
training_epochs = 20
display_step =2

plotdata = {"batchsize": [], "loss": []}

with tf.Session() as sess:          # 好处是可以自动关闭

    sess.run(init)

    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/Users/sheng/Desktop/learn_everyday/tensorflow/log',sess.graph)

    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})     # run 优化器

            summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
            summary_writer.add_summary(summary_str,epoch)
            # tensorboard



        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})       # run cost ,X,Y 指的是可以进行操作的X和Y
            print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    saver.save(sess,"/Users/sheng/Desktop/learn_everyday/tensorflow/tensor-3-1")




#  查看模型内容
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file("/Users/sheng/Desktop/learn_everyday/tensorflow/tensor-3-1",None,True)
# 这个东西就是用来操作的原因了

#





with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,"/Users/sheng/Desktop/learn_everyday/tensorflow/tensor-3-1")


    print("finished")
    print("cost = ",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))

# 图形显示

    plt.plot(train_X,train_Y,'ro',label= "original_data")
    plt.plot(train_X,sess.run(W) * train_X + sess.run(b),label='Fittedline')
    plt.legend()
    plt.show()

    def moving_average(a,w =10):
        if len(a)<w:
            return a[:]
        return [val if idx < w else sum(a[(idx-w):idx]) for idx,val in enumerate(a)]     # 这是要给很好用的东西啊


    plotdata['avgloss'] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],"b--")
    plt.xlabel('Minibatch number')
    plt.ylabel("Loss")
    plt.title("Minibatch run vs Training loss")

    plt.show()








# 在这里面，Sess就像是一个计算板，只要在它的上面进行操作就可以了
# 图是静态的施工图,给了数字才能真正的奔跑起来
# 而OP 代表的是节点，也就是人工智能图中的节点

# 正向我们需要控制
# 而反向，则由图来提供控制

# 1，定义输入节点
# 2. 学习'学习参数'的变量
# 3. 定义运算
# 4. 优化函数
# 5. 初始化变量
# 6.迭代更新
# 7.测试模型
# 8.使用模型
























