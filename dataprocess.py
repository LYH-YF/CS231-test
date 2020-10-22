import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def dataloading():
    data_list=[]
    label_list=[]
    for i  in range(1,6):
        filename="data/cifar-10-batches/data_batch_"+str(i)
        data=unpickle(filename)
        data_list.append(data[b'data'])
        label_list+=data[b'labels']
    datas=np.concatenate(data_list,axis=0)
    return datas,label_list
def testloading():
    filename="data/cifar-10-batches/test_batch"
    data=unpickle(filename)
    return data[b'data'],data[b'labels']
def CIFAR10_TrainData():
    data_list=[]
    label_list=[]
    for i  in range(1,6):
        filename="data/cifar-10-batches/data_batch_"+str(i)
        data=unpickle(filename)
        data_list.append(data[b'data'])
        label_list+=data[b'labels']
    datas=np.concatenate(data_list,axis=0)
    return datas,label_list
def CIFAR10_TestData():
    filename="data/cifar-10-batches/test_batch"
    data=unpickle(filename)
    return data[b'data'],data[b'labels']
def CIFAR10_DataLoading():
    train_data,train_label=CIFAR10_TrainData()
    test_data,test_label=CIFAR10_TestData()
    return train_data,train_label,test_data,test_label
if __name__ == "__main__":
    datas,labels=dataloading()
    train,train_y,test,test_y=CIFAR10_DataLoading()
    print(1)