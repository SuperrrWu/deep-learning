#encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
##定义权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
##定义偏置
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
##定义卷积
def conv2d(x,W):
    #srtide[1,x_movement,y_,movement,1]步长参数说明
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding='SAME')
    #x为输入，W为卷积参数，[5,5,1,32]表示5*5的卷积核，1个channel，32个卷积核。strides表示模板移动步长，SAME和VALID两种形式的padding，valid抽取出来的是在原始图片直接抽取，结果比原始图像小，same为原始图像补零后抽取，结果与原始图像大小相同。
##定义pooling
def max_pool_2x2(x):
    #ksize   strides
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
##定义计算准确度的功能
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result
##输入
xs = tf.placeholder(tf.float32,[None,784])#28*28
ys = tf.placeholder(tf.float32,[None,10])#10个输出
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])
#-1代表样本数不定，28*28大小的图片，1表示通道数
#print(x_image.shape)
##卷积层conv1
W_conv1 = weight_variable([5,5,1,32])#第一层卷积：卷积核大小5x5,1个颜色通道，32个卷积核
b_conv1 = bias_variable([32])#第一层偏置
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#第一层输出：输出的非线性处理28x28x32
h_pool1 = max_pool_2x2(h_conv1)#输出为14x14x32
##卷积层conv2
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)#输出为7x7x64
##全连接层，隐含层的节点个数为1024
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#将2D图像变成1D数据[n_samples,7,7,64]->>[n_samples,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)#非线性激活函数
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)#防止过拟合
##softmaxe层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#输出误差loss,算法cross_entropy+softmax就可以生成分类算法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
#重要的初始化变量操作
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)#从下载好的数据集提取100个数据，mini_batch
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))