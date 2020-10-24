import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def CIFAR10_TrainData_():
    data_list=[]
    label_list=[]
    for i  in range(1,6):
        filename="data/cifar-10-batches/data_batch_"+str(i)
        data=unpickle(filename)
        data_list.append(data[b'data'])
        label_list+=data[b'labels']
    datas=np.concatenate(data_list,axis=0)
    label_list=np.array(label_list)
    return datas,label_list

def CIFAR10_TestData_():
        filename="data/cifar-10-batches/test_batch"
        data=unpickle(filename)
        return data[b'data'],np.array(data[b'labels'])
class CIFAR10_DataLoader(object):
    def __init__(self):
        self.train_inputs,self.train_labels=CIFAR10_TrainData_()
        self.test_inputs,self.test_labels=CIFAR10_TestData_()
        self.train_num=len(self.train_inputs)
        self.test_num=len(self.test_inputs)
    
    def DataLoading(self,batch_size,type):
        inputs=[]
        labels=[]
        loaded_data=[]
        if type=="train":
            inputs=self.train_inputs
            labels=self.train_labels
            max_num=self.train_num
        else:
            inputs=self.test_inputs
            labels=self.test_labels
            max_num=self.test_num
        batch_num=int(max_num/batch_size)+1
        for batch_i in range(batch_num):
            if batch_i+batch_size<=batch_num:
                batch_inputs=inputs[batch_i:batch_i+batch_size]
                batch_labels=labels[batch_i:batch_i+batch_size]
            else:
                batch_inputs=inputs[batch_i:max_num]
                batch_labels=labels[batch_i:max_num]
            if batch_inputs != []:
                loaded_data.append({"inputs":batch_inputs,"labels":batch_labels})
        return loaded_data
if __name__ == "__main__":
    a=np.random.randn(3,4)
    print(a[2:2])