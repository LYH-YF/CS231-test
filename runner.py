import numpy as np
from dataprocess import CIFAR10_DataLoading
from model import NearestNeighbor,NearestNeighbor_torch
import torch
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
def KNNrunner_torch():
    cuda_use= True if torch.cuda.is_available() else False
    
    train_data,train_label,test_data,test_label=CIFAR10_DataLoading()
    train_data=torch.tensor(train_data)
    train_label=torch.tensor(train_label)
    test_data=torch.tensor(test_data)
    test_label=torch.tensor(test_label)
    
    knn_model=NearestNeighbor_torch("L1",100,cuda_use)
    knn_model.train(train_data,train_label)
    y_pred=knn_model.predict(test_data)
    
    result=y_pred==test_label
    acc_rate=sum(result==True).float()/len(result)
    print("acc:",acc_rate)
if __name__ == "__main__":
    #KNNrunner()
    KNNrunner_torch()