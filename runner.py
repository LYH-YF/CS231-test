import numpy as np
from dataprocess import CIFAR10_DataLoading
from model import NearestNeighbor,NearestNeighbor_torch
from loss import Hinge_Loss
from optim import *
import torch
def KNNrunner():
    '''
    run knn model
    '''
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
    '''
    run knn model with gpu(if gpu is available)
    '''
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

def RandomSearchRunner():
    train_data,train_label,test_data,test_label=CIFAR10_DataLoading()
    train_label=np.array(train_label)[:100]
    train_data=train_data[:100]
    test_data=test_data[:100]
    test_label=test_label[:100]
    test_label=np.array(test_label)
    W = np.random.randn(10, 3072)*0.0001
    W=Random_Search(W,train_data,train_label)
    scores=W.dot(test_data.T)
    y_pred=np.argmax(scores,axis=0)
    acc=np.mean(y_pred==test_label)
    print("acc:{}".format(acc))
    
    train_scores=W.dot(train_data.T)
    y_train=np.argmax(train_scores,axis=0)
    train_acc=np.mean(y_train==train_label)
    print("train acc:{}".format(train_acc))
def RandLocalSearchRunner():
    cuda_use=True if torch.cuda.is_available() else False
    train_data,train_label,test_data,test_label=CIFAR10_DataLoading()
    train_label=np.array(train_label)
    train_data=train_data
    test_data=test_data
    test_label=test_label
    test_label=np.array(test_label)
    
    W = np.random.randn(10, 3072)*0.0001
    W=Random_Local_Search(W,train_data,train_label,cuda_use=cuda_use)
    scores=W.dot(test_data.T)
    y_pred=np.argmax(scores,axis=0)
    acc=np.mean(y_pred==test_label)
    print("acc:{}".format(acc))
    
    train_scores=W.dot(train_data.T)
    y_train=np.argmax(train_scores,axis=0)
    train_acc=np.mean(y_train==train_label)
    print("train acc:{}".format(train_acc))
# def RandLocalSearchRunner_torch():
#     cuda_use= True if torch.cuda.is_available() else False
    
#     train_data,train_label,test_data,test_label=CIFAR10_DataLoading()
#     train_data=torch.tensor(train_data[:100]).float()
#     train_label=torch.tensor(train_label[:100]).float()
#     test_data=torch.tensor(test_data[:100]).float()
#     test_label=torch.tensor(test_label[:100]).float()

#     W=torch.randn(10,3072)*0.0001
#     W=Random_Local_Search_torch(W,train_data,train_label,cuda_use=cuda_use)
#     scores=W.dot(test_data.T)
#     y_pred=torch.argmax(scores,axis=0)
#     acc=torch.mean(y_pred==test_label)
#     print("acc:{}".format(acc))
def GradientRunner():
    train_data,train_label,test_data,test_label=CIFAR10_DataLoading()
    train_label=np.array(train_label)[:100]
    train_data=train_data[:100]
    test_data=test_data[:100]
    test_label=test_label[:100]
    test_label=np.array(test_label)
    W = np.random.randn(10, 3072)*0.0001
    Gradient(W,train_data,train_label)
if __name__ == "__main__":
    #KNNrunner()
    #KNNrunner_torch()
    #RandomSearchRunner()
    #RandLocalSearchRunner()
    #RandLocalSearchRunner_torch()
    GradientRunner()