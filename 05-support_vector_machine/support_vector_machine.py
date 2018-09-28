
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from itertools import combinations
import numpy as np


# Implement one-versus-the rest
def ovr(X,y,C):
    ovr_models = []
    for i in C:
        newy = []
        for j in range(len(y)):
            if y[j] == i:
                newy.append(i)
            else:
                newy.append(111)
        clf=SVC(kernel='linear')
        model = clf.fit(X,newy)
        ovr_models.append(model)
    return ovr_models
    # X: input data
    # y: target
    # C: unique set of classes with ascending order
    # return list of binary SVCs     



def ovr_predict(models,X,C):
    pred_y = []
    for i in range(len(models)):
        y=models[i].predict(X)
        pred_y.append(y)
    
    new_pred_y = []
    for j in range(len(X)):
        cls_list = []
        for i in pred_y:
            cls_list.append(i[j])
        cls, counts = np.unique(cls_list,return_counts=True)
        count_cls = np.asarray((cls,counts))
        min_y = count_cls[0][list(count_cls[1]).index(min(count_cls[1]))]
        if min_y == 111:
            min_y = None
        new_pred_y.append(min_y)
    return new_pred_y
    # models: list of binary SVCs 
    # X: input data
    # C: unique set of classes with ascending order
    # return predicted classes of samples in X (if ambiguous, set nan)


# Implement one-versus-one
def ovo(X,y,C):
    index=list(combinations(C,2))
    ovo_models=[]
    for i in range(len(index)):
        index_list=[]
        for j in range(len(y)):
            if y[j] == index[i][0] or y[j] == index[i][1]:
                index_list.append(j)
        newx = X[index_list]
        newy = y[index_list]
        clf=SVC(kernel='linear')
        model = clf.fit(newx,newy)
        ovo_models.append(model)
    return ovo_models
    # X: input data
    # y: target
    # C: unique set of classes with ascending order
    # return list of binary SVCs


def ovo_predict(models,X,C):
    pred_y = []
    for i in range(len(ovo_models)):
        y=ovo_models[i].predict(X)
        pred_y.append(y)
    
    new_pred_y = []
    for j in range(len(X)):
        num_cls = []    
        for i in pred_y:
            num_cls.append(i[j])
        cls, counts = np.unique(num_cls,return_counts=True)
        count_cls = np.asarray((cls,counts))
        max_cls = count_cls[0][list(count_cls[1]).index(max(count_cls[1]))]
        new_pred_y.append(max_cls)
    return new_pred_y
    # X: input data
    # y: target
    # C: unique set of classes with ascending order
    # return predicted classes of samples in X (if ambiguous, set nan)
    


data=datasets.load_digits()
X,y=data.data, data.target
C=np.sort(np.unique(y))
X=scale(X)

Xtrain,Xval,ytrain,yval=train_test_split(X,y,test_size=0.2, random_state=100,stratify=y)

ovr_models=ovr(Xtrain,ytrain,C)
ovr_y_pred=ovr_predict(ovr_models,Xval,C)

ovo_models=ovo(Xtrain,ytrain,C)
ovo_y_pred=ovo_predict(ovo_models,Xval,C)

# compare accuracies for validation set
def acc_com(y_pred, y):
    return sum(y_pred == y)/len(y)

ovr_acc = acc_com(ovr_y_pred, yval)
ovo_acc = acc_com(ovo_y_pred, yval)

print('ovr_acc:{} \novo_acc:{}'.format(ovr_acc, ovo_acc))
