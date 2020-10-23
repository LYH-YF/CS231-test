import numpy as np
import torch
def Hinge_Loss(X,y,W):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    delta=1.0
    scores=W.dot(X.T)
    margin=np.maximum(0,scores-scores[y]+delta)
    loss=margin.sum()-delta
    return loss

def Hinge_Loss_(s,y,delta=1.0):
    margin=np.maximum(0,s-s[y]+delta)
    loss=margin.sum()-delta
    return loss

def Softmax_Loss(s,y):
    return -np.log(s[y])

def eval_numerical_gradient(f, x):
    """  
    一个f在x处的数值梯度法的简单实现
    - f是只有一个参数的函数
    - x是计算梯度的点
    """ 
    fx = f(x) # 在原点计算函数值
    grad = np.zeros(x.shape)
    h = 0.00001
    # 对x中所有的索引进行迭代
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # 计算x+h处的函数值
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # 增加h
        fxh = f(x) # 计算f(x + h)
        x[ix] = old_value # 存到前一个值中 (非常重要)

        # 计算偏导数
        grad[ix] = (fxh - fx) / h # 坡度
        it.iternext() # 到下个维度
    return grad
if __name__ == "__main__":
    a1=np.random.randn(10,3072)
    a2=np.random.randn(3072)
    a3=a1.dot(a2)
    print(a3.shape)
    b1=torch.randn((10,3072))
    b2=torch.randn(20,3072)
    print(b1.shape," ",b2.shape)
    #b3=b1.mul(b2)
    b3=torch.matmul(b1,b2[0])
    print(b3.shape)