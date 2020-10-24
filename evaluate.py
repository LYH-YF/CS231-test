import numpy as np
def cifar_evaluate(W,train_datas,test_datas,train_num,test_num):
    train_acc=0
    for data in train_datas:
        scores=W.dot(data["inputs"].T)
        y_pred=np.argmax(scores,axis=0)
        result=y_pred==data["labels"]
        train_acc+=sum(result==True)
    test_acc=0
    for data in test_datas:
        scores=W.dot(data["inputs"].T)
        y_pred=np.argmax(scores,axis=0)
        result=y_pred==data["labels"]
        test_acc+=sum(result==True)
    train_acc_rate=train_acc/train_num
    test_acc_rate=test_acc/test_num
    print("train_acc [%d/%d]==%.10f"%(train_acc,train_num,train_acc_rate))
    print("test acc  [%d/%d]==%.10f"(test_acc,test_num,test_acc_rate))