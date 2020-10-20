import numpy as np
class NearestNeighbor(object):
    def __init__(self,X,y,p,k) -> None:
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
if __name__ == "__main__":
    a=np.array([1,25,3,6,8,12,7,0,-1,9])
    x=a.argsort()[::-1][0:3]
    print(x)
