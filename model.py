import numpy as np
import math
import torch
from torch import tensor
class NearestNeighbor(object):
    def __init__(self,X,y,p,k):
        self.X=X
        self.y=y
        self.p=p
        self.k=k
    def predict(self,X):
        num_test=X.shape[0]
        y_test=np.zeros(num_test,dtype=self.y.dtype)
        for i in range(num_test):
            print("\rpredict [{}/{}]".format(i+1,num_test),end="")
            if self.p=="L2":
                distence=np.sqrt(np.sum(np.square(self.X-X[i,:]),axis=1))
            else:
                distence=np.sum(np.abs(self.X-X[i,:]),axis=1)
            k_min_idx=distence.argsort()[::-1][:self.k]
            k_min_idx=np.array(k_min_idx)
            labels=self.y[k_min_idx]
            y_test[i]=np.argmax(np.bincount(labels))
            
        return y_test
class NearestNeighbor_torch(object):
    def __init__(self,p,k,cuda_use):
        self.p=p
        self.k=k
        self.cuda_use=cuda_use
    def train(self,X,y):
        self.X=X
        self.y=y
        if self.cuda_use:
            self.X=self.X.cuda()
            self.y=self.y.cuda()
    def predict(self,X):
        if self.cuda_use:
            X=X.cuda()
        num_test=X.shape[0]
        y_test=torch.zeros(num_test,dtype=self.y.dtype)
        for i in range(num_test):
            print("\rpredict [{}/{}]".format(i+1,num_test),end="")
            if self.p=="L2":
                distence=torch.sqrt(torch.sum(torch.square(self.X-X[i,:]),dim=1))
            else:
                distence=torch.sum(torch.abs(self.X-X[i,:]),dim=1)
            k_min_d,k_min_idx=distence.topk(self.k,largest=False)
            #k_min_idx=torch.tensor(k_min_idx)
            labels=self.y[k_min_idx]
            y_test[i]=torch.argmax(torch.bincount(labels))
        return y_test

class Neuron(object):
    # ... 
    def forward(self,inputs):
        """ 假设输入和权重是1-D的numpy数组，偏差是一个数字 """
        cell_body_sum = np.sum(inputs * self.weights) + self.bias
        firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid激活函数
        return firing_rate
if __name__ == "__main__":
    a=tensor([1,25,3,6,8,12,7,0,-1,9])
    b=np.array([1,25,3,6,8,12,7,0,-1,9])
    x,y=a.topk(3,largest=False)
    print(x,y)
    print(a.argsort())
    print(b.argsort())
    
