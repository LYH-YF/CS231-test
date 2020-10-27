import numpy as np
from dataprocess import CIFAR10_DataLoading,CIFAR10_Batch_data,CIFAR10_TestData
from model import *
from loss import Hinge_Loss
from optim import *
from DataLoader import CIFAR10_DataLoader
from evaluate import cifar_evaluate
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

def GradientRunner():
    train_data,train_label,test_data,test_label=CIFAR10_DataLoading()
    train_label=np.array(train_label)
    train_data=train_data
    test_data=test_data
    test_label=test_label
    test_label=np.array(test_label)
    W = np.random.randn(10, 3072)*0.0001
    W=Gradient(W,train_data,train_label)

    train_scores=W.dot(train_data.T)
    y_train=np.argmax(train_scores,axis=0)
    train_acc=np.mean(y_train==train_label)
    print("train acc:{}".format(train_acc))
    scores=W.dot(test_data.T)
    y_pred=np.argmax(scores,axis=0)
    acc=np.mean(y_pred==test_label)
    print("acc:{}".format(acc))
def GradientRunner_(epoch,lr):
    #datas=CIFAR10_Batch_data()
    cifar_dataloader=CIFAR10_DataLoader()
    datas=cifar_dataloader.DataLoading(256,"train")
    W = np.random.randn(10, 3072)*0.0001
    bestW=W
    bestloss=float("inf")
    print("start")
    for epo in range(epoch):
        for step,batch_data in enumerate(datas):
            grad=eval_numerical_gradient(W,batch_data["inputs"],batch_data["labels"])
            W=W-grad*lr
            loss=Hinge_L(batch_data["inputs"],batch_data["labels"],W)
            print("epoch:[%d/%d] step[%d]:loss[%.8f]"%(epo+1,epoch,step+1,loss))
            if loss<bestloss:
                bestW=W
                bestloss=loss
    test_datas=cifar_dataloader.DataLoading(256,"test")
    cifar_evaluate(bestW,
                    datas,
                    test_datas,
                    cifar_dataloader.train_num,
                    cifar_dataloader.test_num)
def GradientRunner_torch(epoch,lr):
    cuda_use= True if torch.cuda.is_available() else False
    #datas=CIFAR10_Batch_data()
    #W = np.random.randn(10, 3072)*0.0001
    W=torch.randn(10,3072)*0.0001
    if cuda_use:
            W=W.cuda()
    bestW=W
    bestloss=float("inf")
    for epo in range(epoch):
        step=0
        for batch_data in CIFAR10_Batch_data():
            step+=1
            #grad=eval_numerical_gradient(W,batch_data["input"],batch_data["target"])
            X_train=torch.tensor(batch_data["input"]).float()
            y_train=torch.tensor(batch_data["target"])
            grad=Eval_Gradient(W,X_train,y_train,cuda_use=cuda_use)
            W=W-grad*lr
            #loss=Hinge_L(batch_data["input"],batch_data["target"],W)
            loss=Hinge_L_torch(X_train,y_train,W,cuda_use=cuda_use)
            print("epoch:[%d/%d] step[%d]:loss[%.8f]"%(epo,epoch,step,loss))
            if loss<bestloss:
                bestW=W
                bestloss=loss
        print()
    test_data,test_label=CIFAR10_TestData()
    if cuda_use:
        test_data=torch.tensor(test_data).float().cuda()
        test_label=torch.tensor(test_label).cuda()
    else:
        test_data=torch.tensor(test_data).float()
        test_label=torch.tensor(test_label)
    #scores=bestW.dot(test_data.T)
    scores=torch.matmul(bestW,test_data.T)
    #y_pred=np.argmax(scores,axis=0)
    y_pred=torch.argmax(scores,axis=0)
    #acc=np.mean(y_pred==test_label)
    result=y_pred==test_label
    acc=sum(result==True).float()/len(result)
    #acc=torch.mean(y_pred==test_label)
    print("acc:{}".format(acc))
def Softmax_Runner(epoch,lr):
    cifar_dataloader=CIFAR10_DataLoader()
    datas=cifar_dataloader.DataLoading(256,"train")
    test_datas=cifar_dataloader.DataLoading(256,"test")
    model=Softmax_Classifier(3072,10)
    bestW=model.W
    bestloss=float("inf")
    print("start train")
    for epo in range(epoch):
        for step,batch_data in enumerate(datas):
            model.train(batch_data["inputs"],batch_data["labels"],Softmax_Loss,lr)
            loss=model.eval(batch_data["inputs"],batch_data["labels"],Softmax_Loss)
            print("epoch:[%d/%d] step[%d]:loss[%.8f]"%(epo+1,epoch,step+1,loss))
            cifar_evaluate(W=model.W,
                            test_datas=test_datas,
                            test_num=cifar_dataloader.test_num)
            if loss<bestloss:
                bestW=W
                bestloss=loss
    print("start test")
    cifar_evaluate(bestW,
                    datas,
                    test_datas,
                    cifar_dataloader.train_num,
                    cifar_dataloader.test_num)
if __name__ == "__main__":
    #KNNrunner()
    #KNNrunner_torch()
    #RandomSearchRunner()
    #RandLocalSearchRunner()
    #RandLocalSearchRunner_torch()
    #GradientRunner_(200,0.001)
    #GradientRunner_torch(200,0.001)
    Softmax_Runner(10,0.001)