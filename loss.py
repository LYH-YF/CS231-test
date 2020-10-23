import numpy as np
import torch
def Hinge_Loss(X,y,W):
    delta=1.0
    scores=W.dot(X.T)
    margin=np.maximum(0,scores-scores[y]+delta)
    loss=margin.sum()-delta
    return loss

def Hinge_Loss_(s,y,delta=1.0):
    margin=np.maximum(0,s-s[y]+delta)
    loss=margin.sum()-delta
    return loss

def Hinge_L(X,y,W):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    delta=1.0
    scores=np.matmul(X,W.T)
    row_idx=np.arange(scores.shape[0])
    margin=np.maximum(0,scores-np.matrix(scores[row_idx,y]).T+delta)
    loss=margin.sum()-delta
    return loss
def Softmax_Loss(s,y):
    return -np.log(s[y])

if __name__ == "__main__":
    # b1=torch.randn((10,3072))
    # b2=torch.randn(20,3072)
    # y=torch.randint(low=0,high=9,size=(20,1)).squeeze()
    # score=torch.matmul(b1,b2.T)
    # aa=score[y]
    # print(b1.shape," ",b2.shape)
    a1=np.random.randn(10,3072)
    a2=np.random.randn(20,3072)
    y=np.random.randint(0,9,size=(20))
    score=np.matmul(a1,a2.T)
    aa=score[y]
    print(1)