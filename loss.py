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
    loss=margin.sum()-delta*y.shape[0]
    return loss
def Hinge_L_torch(X,y,W,cuda_use=False):
    if cuda_use:
        X=X.cuda()
        W=W.cuda()
    delta=1.0
    scores=torch.matmul(W,X.T)
    row_idx=torch.arange(scores.shape[1])
    margin=scores-scores[y,row_idx]+delta
    zero=torch.tensor([0.])
    margin=torch.max(zero,margin)
    loss=margin.sum()-delta*y.shape[0]
    return loss
def Softmax_Loss(s,y):
    return -np.log(s[y])

if __name__ == "__main__":
    X=torch.randn((2,10))
    W=torch.randn(10,10)
    y=torch.randint(0,9,size=(2,1)).squeeze()
    print(Hinge_L_torch(X,y,W))