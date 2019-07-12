import numpy as np
class Network():
    def __init__(self,sizes):#构造函数
        self.num_layers=len(sizes)#读出有几层神经网络
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]#sizes[1:]除去list的第一个值（3，1）
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

net=Network([2,3,1])#每层神经元的个数
print(net.biases)
print(net.weights)