import numpy as np

def Sigmoid(X):
    return 1.0/(1+np.exp(-X))
def Tanh(X):
    return (1-np.exp(-2*X))/(1+np.exp(-2*X))
def ReLU(X):
    return np.maximum(0,X)
def LeakyReLU(X):
    pass
def Maxout():
    pass

if __name__ == "__main__":
    a=np.random.randn(2,3)
    print(Sigmoid(a))
    print(Tanh(a))
    print(ReLU(a))