
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


def create_bootstrap(X,y,ratio):
    sample_set=list(np.random.randint(0,len(X),size = int(len(X)*ratio)).reshape(-1))
    newX = X[sample_set]
    newy = y[sample_set]
    return newX, newy

    # X: input data matrix
    # ratio: sampling ratio
    # return bootstraped dataset (newX,newy)
    

def voting(y):
    max_num=[]
    for i in range(len(y)):
        num, counts = np.unique(y[i], return_counts=True)
        num_list = np.asarray((num, counts))
        max_list = num_list[0][list(num_list[1]).index(max(num_list[1]))]
        max_num.append(max_list)
    return np.array(max_num)
      
    # y: 2D matrix with n samples by n_estimators
    # return voting results by majority voting (1D array)



# bagging
def bagging_cls(X,y,n_estimators,k,ratio):
    knn_models = []
    for i in range(n_estimators):
        newX, newy = create_bootstrap(X,y,ratio)
        knn_M = KNeighborsClassifier(n_neighbors=k)
        knn_M.fit(newX,newy)
        knn_models.append(knn_M)
    return knn_models
    
    # X: input data matrix
    # y: output target
    # n_estimators: the number of classifiers
    # k: the number of nearest neighbors
    # ratio: sampling ratio
    # return list of n k-nn models trained by different boostraped sets
    
    
data=load_iris()
X=data.data[:,:2]
y=data.target    

n_estimators=3
k=3
ratio=0.8

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = np.c_[xx.ravel(), yy.ravel()]

models = bagging_cls(X,y,n_estimators,k,ratio)
y_models = np.zeros((len(xx.ravel()),n_estimators))
for i in range(n_estimators):
    y_models[:,i]=models[i].predict(Z)

y_pred=voting(y_models)

# Draw decision boundary
fig = plt.subplot()
plt.contourf(xx,yy,y_pred.reshape(xx.shape),cmap=plt.cm.RdYlBu,alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y, s =30)
plt.show()
