
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def create_bootstrap(X,y,ratio):
    ind=list(np.random.randint(0,len(X),size = int(len(X)*ratio)).reshape(-1))
    newX = X[ind]
    newy = y[ind]
    return newX, newy, ind
    # X: input data matrix
    # ratio: sampling ratio
    # return one bootstraped dataset and indices of sub-sampled samples (newX,newy,ind)


def cal_oob_error(X,y,models,ind):
    oob_errors = []
    for i in range(500):
        X2 = np.delete(X,ind[i],axis=0)
        y2 = np.delete(y,ind[i],axis=0)
        oob = 1-models[i].score(X2,y2)
        oob_errors.append(oob)
    return oob_errors
    # X: input data matrix
    # y: y: output target
    # models: list of trained models by different bootstraped sets
    # ind: list of indices of samples in different bootstraped sets
    

def cal_var_importance(X,y,models,ind,oob_errors):
    avg_oob = np.mean(oob_errors)
    var_imp = []
    for i in range(30):
        X[i] = np.random.permutation(X[i])
        error = []
        for j in range(500):
            X3 = np.delete(X,ind[i],axis=0)
            y3 = np.delete(y,ind[i],axis=0)
            oob2 = 1-models[j].score(X3,y3)
            error.append(oob2)
        importance = np.mean(error) - avg_oob
        var_imp.append(importance)
    return var_imp
    # X: input data matrix
    # y: output target
    # models: list of trained models by different bootstraped sets
    # ind: list of indices of samples in different bootstraped sets
    # oob_errors: list of oob error of each sample
    # return variable importance
    
    

def random_forest(X,y,n_estimators,ratio,params):
    models = []
    ind_set = []
    tree = DecisionTreeClassifier(**params)
    for i in range(n_estimators):
        newX, newy, ind=create_bootstrap(X,y,ratio)
        models.append(tree.fit(newX, newy))
        ind_set.append(ind)
    return models, ind_set
    # X: input data matrix
    # y: output target
    # n_estimators: the number of classifiers
    # ratio: sampling ratio for bootstraping
    # params: parameter setting for decision tree
    # return list of tree models trained by different bootstraped sets and list of indices of samples in different bootstraped sets
    # (models,ind_set)



data=datasets.load_breast_cancer()
X, y = shuffle(data.data, data.target, random_state=13)

params = {'max_depth': 4, 'min_samples_split': 0.1, 'min_samples_leaf':0.05}
n_estimators=500
ratio=1.0

models, ind_set = random_forest(X,y,n_estimators,ratio,params)
oob_errors=cal_oob_error(X,y,models,ind_set)
var_imp=cal_var_importance(X,y,models,ind_set,oob_errors)

nfeature=len(X[0])
plt.barh(np.arange(nfeature),var_imp/sum(var_imp))
plt.yticks(np.arange(nfeature) + 0.35 / 2, data.feature_names)
