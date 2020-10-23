from loss import *
import numpy as np
import torch
def Random_Search(W,X_train,Y_train):
    # 假设X_train的每一列都是一个数据样本（比如3073 x 50000）
    # 假设Y_train是数据样本的类别标签（比如一个长50000的一维数组）
    # 假设函数L对损失函数进行评价
    bestW=W
    bestloss = float("inf") # Python assigns the highest possible float value
    for num in range(1000):
        loss_total=0.0
        W = np.random.randn(10, 3072) * 0.0001 # generate random parameters
        for idx,data in enumerate(X_train):
            #loss = L(X_train, Y_train, W) # get the loss over the entire training set
            loss_total+=Hinge_Loss(data,Y_train[idx],W)
        if loss_total < bestloss: # keep track of the best solution
            bestloss = loss_total
            bestW = W
        print('in attempt [%d] the loss was %f, best %.10f' % (num, loss_total, bestloss))
            #print("\repoch[{}/1000] step[{}]:loss[{}] bestloss[{}]".format(num,idx,loss,bestloss),end="")
    
    return bestW

def Random_Local_Search(W,X_train,Y_train,step_size=0.0001):
    bestloss = float("inf") # Python assigns the highest possible float value
    for num in range(1000):
        loss_total=0.
        Wtry = W + np.random.randn(10, 3072) * step_size
        for idx,data in enumerate(X_train):
            #loss = L(Xtr_cols, Ytr, Wtry)
            loss_total+=Hinge_Loss(data,Y_train[idx],Wtry)
        if loss_total < bestloss:
            W = Wtry
            bestloss = loss_total
        print ('iter [%d/1000] loss is [%f]' % (num, bestloss))
    return W
def Random_Local_Search_torch(W,X_train,Y_train,step_size=0.0001,cuda_use=False):
    bestloss=0.
    if cuda_use:
        W=W.cuda()
        X_train=X_train.cuda()
    for num in range(1000):
        loss_total=0.
        delta_W=torch.randn(10, 3072) * step_size
        if cuda_use:
            delta_W=delta_W.cuda()
        Wtry = W + delta_W
        for idx,data in enumerate(X_train):
            #loss = L(Xtr_cols, Ytr, Wtry)
            #loss_total+=Hinge_Loss(data,Y_train[idx],Wtry)
            score=Wtry.matmul(data)
            loss_total+=Hinge_Loss_(score,Y_train[idx])
        if loss_total < bestloss:
            W = Wtry
            bestloss = loss_total
        print ('iter [%d/1000] loss is [%f]' % (num, bestloss))
    return W
if __name__ == "__main__":
    pass