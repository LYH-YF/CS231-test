from loss import *
import numpy as np
import torch
# def Random_Search(W,X_train,Y_train):
#     # 假设X_train的每一列都是一个数据样本（比如3073 x 50000）
#     # 假设Y_train是数据样本的类别标签（比如一个长50000的一维数组）
#     # 假设函数L对损失函数进行评价
#     bestW=W
#     bestloss = float("inf") # Python assigns the highest possible float value
#     for num in range(1000):
#         loss_total=0.0
#         W = np.random.randn(10, 3072) * 0.0001 # generate random parameters
#         for idx,data in enumerate(X_train):
#             #loss = L(X_train, Y_train, W) # get the loss over the entire training set
#             loss_total+=Hinge_Loss(data,Y_train[idx],W)
#         if loss_total < bestloss: # keep track of the best solution
#             bestloss = loss_total
#             bestW = W
#         print('in attempt [%d] the loss was %f, best %.10f' % (num, loss_total, bestloss))
#             #print("\repoch[{}/1000] step[{}]:loss[{}] bestloss[{}]".format(num,idx,loss,bestloss),end="")
    
#     return bestW
def Random_Search(W,X_train,Y_train):
    # 假设X_train的每一列都是一个数据样本（比如3073 x 50000）
    # 假设Y_train是数据样本的类别标签（比如一个长50000的一维数组）
    # 假设函数L对损失函数进行评价
    bestW=W
    bestloss = float("inf") # Python assigns the highest possible float value
    for num in range(1000):
        loss_total=0.0
        W = np.random.randn(10, 3072) * 0.0001 # generate random parameters
        loss_total=Hinge_L(X_train,Y_train,W)
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

def eval_numerical_gradient(W,X_train,y_train):
    """  
    一个f在x处的数值梯度法的简单实现
    - f是只有一个参数的函数
    - x是计算梯度的点
    """ 
    #fx = f(x) # 在原点计算函数值
    fx=Hinge_L(X_train,y_train,W)
    grad = np.zeros(W.shape)
    h = 0.00001
    # 对x中所有的索引进行迭代
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # 计算x+h处的函数值
        ix = it.multi_index
        old_value = W[ix]
        W[ix] = old_value + h # 增加h
        #fxh = f(x) # 计算f(x + h)
        fxh=Hinge_L(X_train,y_train,W)
        W[ix] = old_value # 存到前一个值中 (非常重要)
        # 计算偏导数
        grad[ix] = (fxh - fx) / h # 坡度
        it.iternext() # 到下个维度
    return grad
def Eval_Gradient(W,X_train,y_train,cuda_use=False):
    if cuda_use:
        W=W.cuda()
        X_train=X_train.cuda()
    fx=Hinge_L_torch(X_train,y_train,W,cuda_use=cuda_use)
    grad=torch.zeros(W.shape)
    h = 0.00001
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            old_value = W[i,j]
            W[i,j] = old_value + h
            fxh=Hinge_L_torch(X_train,y_train,W,cuda_use=cuda_use)
            W[i,j]=old_value
            grad[i,j]=(fxh-fx)/h
    if cuda_use:
        grad=grad.cuda()
    return grad
def Gradient(W,X_train,y_train,step=0.001):
    i=0
    while True:
        i+=1
        if i>200:
            break
        grad=eval_numerical_gradient(W,X_train,y_train)
        W=W-grad*step
        loss=Hinge_L(X_train,y_train,W)
        print("\riter [%d] :%.10f "%(i,loss))
        if loss<0.01:
            break
    return W
if __name__ == "__main__":
    W = np.random.randn(5, 10)
    X = np.random.randn(10, 3)
    D = W.dot(X)
    print(1)