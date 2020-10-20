import numpy as np
from dataprocess import CIFAR10_DataLoading
from model import NearestNeighbor
def KNNrunner():
    train_data,train_label,test_data,test_label=CIFAR10_DataLoading()
    train_label=np.array(train_label)

    test_data=test_data[:100]
    test_label=test_label[:100]
    test_label=np.array(test_label)
    knn_model=NearestNeighbor(train_data,train_label,"L1",100)
    #L1 20 0.31
    #L1 50 0.31
    #L1 100 0.33
    #L2 20 0.16
    #L2 10 0.18
    #L2 5  0.2
    #L2 1  0.16
    y_pred=knn_model.predict(test_data)

    result=y_pred==test_label
    acc_rate=sum(result==True)/len(result)
    print("acc:",acc_rate)
if __name__ == "__main__":
    KNNrunner()