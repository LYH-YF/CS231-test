import numpy as np
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
    '''
    # 假设X_train的每一列都是一个数据样本（比如3073 x 50000）
    # 假设Y_train是数据样本的类别标签（比如一个长50000的一维数组）
    # 假设函数L对损失函数进行评价

    bestloss = float("inf") # Python assigns the highest possible float value
    for num in xrange(1000):
    W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
    loss = L(X_train, Y_train, W) # get the loss over the entire training set
    if loss < bestloss: # keep track of the best solution
        bestloss = loss
        bestW = W
    print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

    # 输出:
    # in attempt 0 the loss was 9.401632, best 9.401632
    # in attempt 1 the loss was 8.959668, best 8.959668
    # in attempt 2 the loss was 9.044034, best 8.959668
    # in attempt 3 the loss was 9.278948, best 8.959668
    # in attempt 4 the loss was 8.857370, best 8.857370
    # in attempt 5 the loss was 8.943151, best 8.857370
    # in attempt 6 the loss was 8.605604, best 8.605604
    # ... (trunctated: continues for 1000 lines)
    #在上面的代码中，我们尝试了若干随机生成的权重矩阵W，其中某些的损失值较小，而另一些的损失值大些。我们可以把这次随机搜索中找到的最好的权重W取出，然后去跑测试集：

    # 假设X_test尺寸是[3073 x 10000], Y_test尺寸是[10000 x 1]
    scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
    # 找到在每列中评分值最大的索引（即预测的分类）
    Yte_predict = np.argmax(scores, axis = 0)
    # 以及计算准确率
    np.mean(Yte_predict == Yte)
    # 返回 0.1555
    '''