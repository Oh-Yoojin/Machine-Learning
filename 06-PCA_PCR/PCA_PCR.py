
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def cal_PC(X, n_components):
    for i in X.T:
        i -= np.mean(i)
    cov_x = np.cov(X.T)
    eigval_list,eigvec_list = np.linalg.eig(cov_x)
    eigval=eigval_list[:n_components]
    eigvec = []
    for i in range(n_components):
        vec = eigvec_list[:,i]
        eigvec.append(vec)
    eigvec=np.asarray(eigvec)
    return eigval, eigvec
    
    # X: input data matrix
    # n_components: the number of principal components
    # return (eigenvalues of n_components of PCs, n_feature*n_components matrix (each column is PC))




def proj_PC(X,eigvec):
    proj_X=np.matmul(X,eigvec.T)
    return proj_X
    # X: input data matrix
    # eigvec: n_feature*n_components matrix (each column is PC)
    # return n_data*n_components transformed data matrix


    
def PCR(X, y, n_components):
    reg = LinearRegression()
    eigval_R, eigvec_R = cal_PC(X,n_components)
    new_X = proj_PC(X,eigvec_R)
    reg_model = reg.fit(new_X,y)
    return reg_model
    # X: input data matrix
    # y: output target vector
    # n_components: the number of principal components
    # return regression model

# PCA
iris=datasets.load_iris()
X1=iris.data
y1=iris.target
n_components=2

eigval,eigvec=cal_PC(X1, n_components)
T1=proj_PC(X1, eigvec)

# TODO: Get transformed data using PCA implemented by scikit-learn
pca=PCA(n_components=2)
pca.fit(X1)
y_pca=pca.transform(X1)

# TODO: Plot 
plt.scatter(y_pca[:,0],y_pca[:,1],c=y1)
plt.scatter(T1[:,0], T1[:,1], c=y1)

# Regression
n_components=4
boston=datasets.load_boston()
X2=boston.data
y2=boston.target
reg_pca=PCR(X2,y2,n_components)

# TODO: Build a regression model using all features
reg = LinearRegression()
reg.fit(X2, y2)
reg.score(X2, y2)

# TODO: Compare R-square using all samples of PCR with ordinary regression model
reg_pcr=PCR(X2,y2,n_components)
eigval,eigvec=cal_PC(X2, n_components)
T1=proj_PC(X2, eigvec)
reg_pcr.score(T1, y2)

