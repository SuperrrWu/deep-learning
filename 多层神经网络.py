# encoding:utf-8
# add_layer_1.py 练习增加网络层

# 创建神经网络并训练
# 功能：拟合 y=x*x+1

import tensorflow as tf
import numpy as np
LEARNING_RATE_BASE=0.1
LEARNING_RATE_DECAY=0.9
LEARNING_RATE_STEP=1

# 创建一个神经网络层
def add_layer(input, in_size, out_size, activation_function=None):
    """
    :param input: 数据输入
    :param in_size: 输入大小
    :param out_size: 输出大小
    :param activation_function: 激活函数（默认没有）
    :return:output：数据输出
    """
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    W_mul_x_plus_b = tf.matmul(input, Weight) + biases
    # 根据是否有激活函数
    if activation_function == None:
        output = W_mul_x_plus_b
    else:
        output = activation_function(W_mul_x_plus_b)
    return output


# 创建一个具有输入层，隐藏层，输出层的三层神经网络，神经元个数分别为1，10，1
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 创建输入数据  np.newaxis分别是在列(第二维)上增加维度，原先是（300，）变为（300，1）
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + 1 + noise  # 创建输入数据对应的输出
#print(x_data)
#print(y_data)
# print(x_data.shape)
# x=np.linspace(-1,1,300)
# print(x.shape)

# 定义输入数据
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义一个隐藏层
hidden_layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 定义一个输出层
prediction = add_layer(hidden_layer1, 10, 1, activation_function=None)

# 求解神经网络参数
# 1.
#loss = tf.reduce_sum(tf.square(prediction - ys))  # sum of the squares
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction) ,reduction_indices=[1] ))
# 2.定义训练过程
global_step=tf.Variable(0,trainable=False)

learning_rate=tf.train.exponential_decay(

LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True

)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 3.进行训练
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    print("第"+str(i)+"轮：",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

    #if i % 100 == 0:
    #    print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

# 关闭sess
sess.close()
