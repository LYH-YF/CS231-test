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
def CIFAR10_Batch_data():
    for i  in range(1,6):
        filename="data/cifar-10-batches/data_batch_"+str(i)
        data=unpickle(filename)
        inputs=data[b'data']
        labels=np.array(data[b'labels'])
        yield {"input":inputs,"target":labels}

def PCA(X):
    X -= np.mean(X, axis = 0) # 对数据进行零中心化(重要)
    cov = np.dot(X.T, X) / X.shape[0] # 得到数据的协方差矩阵
    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X,U) # 对数据去相关性
    Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced 变成 [N x 100]
    return Xrot_reduced
def Whitening(X):
    X -= np.mean(X, axis = 0) # 对数据进行零中心化(重要)
    cov = np.dot(X.T, X) / X.shape[0] # 得到数据的协方差矩阵
    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X,U) # 对数据去相关性
    # 对数据进行白化操作:
    # 除以特征值 
    Xwhite = Xrot / np.sqrt(S + 1e-5)
    return Xwhite
if __name__ == "__main__":
    train,train_y,test,test_y=CIFAR10_DataLoading()
    train_pca=PCA(np.float64(train))
    train_white=Whitening(np.float64(train))
    print(1)