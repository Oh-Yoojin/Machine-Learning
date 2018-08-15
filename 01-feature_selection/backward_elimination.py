
# Use following packages only 
import pandas as pd
from sklearn import linear_model
import scipy

# load data
df=pd.read_csv(r'https://o365seoultech-my.sharepoint.com/personal/kyoungok_kim_office_seoultech_ac_kr/_layouts/15/guestaccess.aspx?docid=1be27e120c44a43c8a60b028fc2bea1d9&authkey=AYBDN_VSuQFVdF7wHBiaz30', sep='\t', names=['aluminate','silcate','ferrite','dical_sil','hardening'], skiprows=1)
p_remove = 0.15

# 초기값 정의
x = df[['aluminate','silcate','ferrite','dical_sil']]
y = df['hardening']

clf = linear_model.LinearRegression()
clf.fit(x,y)

n = len(y)
p = len(x.iloc[1])
y_mean = sum(y)/n
y_pred = clf.predict(x)
SSE = sum((y_pred-y)**2)
MSE = SSE/(n-p-1)
SSR_t = sum((y_pred-y_mean)**2)

# backward elimination
eliminated = []
while True:
    SSR_list = []
    f_p_value = []

    for i in x:
        candidate = x.drop(i, axis=1)
        clf.fit(candidate,y)
        p = len(candidate.iloc[0])
        y_pred = clf.predict(candidate)
        SSE = sum((y_pred-y)**2)
        MSE = SSE/(n-p-1)
        SSR = sum((y_pred-y_mean)**2)
        SSR_list.append(SSR)
        partial_SSR = SSR_t-SSR
        partial_f = partial_SSR/MSE
        f_p_value.append(1-scipy.stats.f.cdf(partial_f,1,n-p-1))

    if max(f_p_value) > p_remove:
        name = x.columns[f_p_value.index(max(f_p_value))]
        eliminated.append(name)
        drop_columns = x.columns[f_p_value.index(max(f_p_value))]
        x = x.drop(drop_columns, axis=1)
        SSR_t = SSR_list[f_p_value.index(max(f_p_value))]
    
    else:
        
        print("Eliminated variable : {}".format(drop_columns))
        break
    
    
