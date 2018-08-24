
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# implement consine kernel
def cosine_kernel(x, train, h):
    y=0
    for i in train:
        if abs((i-x)/h) <= 1:
            y = y + np.pi/4*np.cos((np.pi/2)*((i-x)/h))
    pw = y/(len(train)*h)
    return pw
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)


# implement gaussian kernel
def gaussian_kernel(x, train, h):
    y=0
    for i in train:
        y = y + np.exp((-(i-x)/h).T*((i-x)/h)/2)*(1/np.sqrt(2*np.pi))
    pw = y/(len(train)*h)
    return pw
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
    
# implement 2D gaussian kernel
def gaussian_2d_kernel(x, train, h, sig):
    y = 0
    for i in train:
        u1 = (i[0]-x[0])/h
        u2 = (i[1]-x[1])/h
        sqrt1 = np.sqrt(2*np.pi*(sig[0]))
        sqrt2 = np.sqrt(2*np.pi*(sig[1]))
        try1 = np.exp((-u1.T*u1)/(2*sig[0]))
        try1_1 = try1*(1/sqrt1)
        try2 = np.exp((-u2.T*u2)/(2*sig[1]))
        try2_1 = try2*(1/sqrt2)
        y = y + (try1_1 * try2_1)
    pw = y/(len(train)*h)
    return pw
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)



# implement epanechnikov kernel
def epanechnikov_kernel(x,train,h):
    y=0
    for i in train:
        if abs((i-x)/h) <= 1:
            y = y + 3/4*(1-((i-x)/h)**2)
    pw = y/(len(train)*h)
    return pw
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    

    
def kde1d(train,test,kernel,h):
    # train: training set
    # test: test set
    # kernel: kernel function (object)
    # h: bandwidth
    
    d = [kernel(x,train,h) for x in test]
    return d

def kde2d(train,test,kernel,h,sig):
    # train: training set
    # test: test set
    # kernel: kernel function (object)
    # h: bandwidth
    
    d = [kernel(x, train, h,sig) for x in test]
    return d

# 1D
sample=[2,3,4,8,10,11,12]
h=1
x=np.linspace(0,14,100000)

y1=kde1d(sample,x,cosine_kernel,h)
y2=kde1d(sample,x,gaussian_kernel,h)
y3=kde1d(sample,x,epanechnikov_kernel,h)
    
fig=plt.subplots(1,3,figsize=(10,4))
plt.subplot(1,3,1)
plt.plot(x,y1)
plt.title('Cosine')
plt.subplot(1,3,2)
plt.plot(x,y2)
plt.title('Gaussian')
plt.subplot(1,3,3)
plt.plot(x,y3)
plt.title('Epanechnikov')
plt.show()

#2D
sample_2d=pd.read_table(r'https://o365seoultech-my.sharepoint.com/personal/kyoungok_kim_office_seoultech_ac_kr/_layouts/15/guestaccess.aspx?docid=13fe58da725a54fbd8b1b4b838aa74f38&authkey=AUeuJ5g6Lbiug1gRzPo4Ahg')
h=1
sum_stats=sample_2d.describe()
xmin,ymin=sum_stats.loc['min']-0.5
xmax,ymax=sum_stats.loc['max']+0.5

x=np.linspace(xmin,xmax,100)
y=np.linspace(ymin,ymax,100)
X,Y=np.meshgrid(x,y)
Z = np.c_[X.ravel(),Y.ravel()]

Z1 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,h,[0.5,0.5])
Z1 = np.reshape(Z1, X.shape)
Z2 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,h,[1,1])
Z2 = np.reshape(Z2, X.shape)
Z3 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,h,[2,2])
Z3 = np.reshape(Z3, X.shape)

fig,ax=plt.subplots(1,3,figsize=(16,4))
plt.subplot(1,3,1)
cs=plt.contourf(X,Y,Z1,cmap=plt.cm.Blues)
plt.colorbar(cs)
plt.subplot(1,3,2)
cs=plt.contourf(X,Y,Z2,cmap=plt.cm.Blues)
plt.colorbar(cs)
plt.subplot(1,3,3)
cs=plt.contourf(X,Y,Z3,cmap=plt.cm.Blues)
plt.colorbar(cs)
plt.show()
