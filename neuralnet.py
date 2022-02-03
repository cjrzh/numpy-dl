import pandas as pd
import numpy as np
'''
Feedforward neural network with numpy
用numpy手撸神经网络
'''

class NeuralNet():
    
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    #最后一维等于类别数
    @staticmethod
    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True) #防止数值溢出
        e_x=np.exp(x)
        return e_x/np.sum(e_x,axis=-1,keepdims=True)

    def __init__(self) -> None:
        
        ## init中一般初始化参数
        self.labels = None
        
        # 输入层784，第一层50 第二层100，输出层10
        # 更robust的方法是将层数作为参数传进来
        # 作为演示，直接硬编码
        #初始化参数
        self.parameters = {            
            "W1":np.random.randn(784,50),
            "b1":np.random.randn(1,50),
            "W2":np.random.randn(50,100),
            "b2":np.random.randn(1,100),
            "W3":np.random.randn(100,10),
            "b3":np.random.randn(1,10),
        }
    
    def feedforward(self,X):
        P = self.parameters
        
        a1=np.dot(X,P['W1'])+P['b1']
        Z1=self.sigmoid(a1)
        a2=np.dot(Z1,P['W2'])+P['b2']
        Z2=self.sigmoid(a2)
        a3=np.dot(Z2,P['W3'])+P['b3']
        y=self.softmax(a3)
        return np.argmax(y,axis=1)

