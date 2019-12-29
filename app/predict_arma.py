
import numpy as np
import statistics as st

#from statsmodels.tsa.stattools import pacf
#import matplotlib.pyplot as plt

PCF = 0.2
LEN_PCF = 10
'''
endog = [6.2305232e+000,
8.9768331e+000,
9.6843221e+000,
9.8633070e+000,
1.0037810e+001,
1.0136458e+001,
9.9455382e+000,
9.8584586e+000,
1.0239049e+001,
1.0550768e+001,
1.0582943e+001,
1.0688708e+001,
1.0968963e+001,
1.1122068e+001,
1.1127740e+001,
1.1162554e+001,
1.1223776e+001,
1.1071565e+001,
1.0832010e+001,
1.0661435e+001,
1.0554111e+001,
1.0619270e+001,
1.0693566e+001,
1.0634495e+001,
1.0577182e+001,
1.0594517e+001,
1.0596175e+001,
1.0550704e+001,
1.0507293e+001,
1.0474696e+001,
1.0389663e+001,
1.0268343e+001,
1.0150247e+001,
1.0048013e+001,
1.0065705e+001,
1.0120748e+001,
1.0131985e+001,
1.0119168e+001,
1.0070323e+001,
1.0014623e+001,
9.9610537e+000,
9.9448484e+000,
9.9862680e+000,
1.0013852e+001,
1.0015691e+001,
1.0030285e+001,
1.0056620e+001,
1.0080256e+001]
'''


def acf(y):
    y = np.array(y)
    m = np.mean(y)
    var = st.variance(y)
    r = []
    n = len(y)
    for s in range(n-1):
        r.append(np.sum( (y[s+1:] - m) * (y[:n-s-1] - m) ))
    r = np.array(r)
    return r*(1/( (n-1)* (var)))


#print(acf(x)) #[ 0.25 -0.3  -0.45]
def pacf(y):
    r = acf(y)
    r = np.append(r, 0.0)
    y = np.array(y)
    n = len(y)
    f = np.zeros(shape = (n, n), dtype=float)
    f[0,0] = r[0]
    for k in range(1,n):
        sum1 = sum2 = 0
        for j in range(k):
            if k-1 != j :
                f[k-1, j] = f[k-2, j] - f[k-1, k-1] * f[k-2, k-2-j]
            sum1 +=  f[k-1, j]*r[k-j-1]
        f[k, k] = (r[k] - sum1)/(1- np.sum(f[k-1, :k] * r[:k]))
    #print(f)
    pacf = np.array([f[i,i] for i in range(n)])
    return pacf[:-1]
 #[0.25, -0.38666666666666671, -0.31270903010033446]


def calc_a(endog, order):
    n = len(endog)
    a = np.zeros(shape = (n - 1, order[-1]+1), dtype = float)
    a[:,0]=1
    b = endog[1:]
    for j in range(1,order[-1]+1):
        for i in range(n-1):
            if i-j >= -1:
                a[i,j] = endog[i-j+1]
    print(a)
    a = a[:, np.insert(order, 0, 0)] # delete not use lags: order = [2,4] => delete 1, 3 column//not use sthis lags
    x = np.linalg.lstsq(a,b)[0] #our a: y(n) =a0+a1*y(n-1)+a2*y(n-2)
    print(x, a)
    #return x, b - a*x.T


def ar(endog, forecast):
    n = len(endog)
    endog = np.array(endog)
    if st.variance(endog) ==0:
        return st.mean(endog)*np.ones(forecast)
    pacf_endog = pacf(endog)
    print('pacf', pacf_endog)
    order = np.where(abs(pacf_endog)>PCF)[0]+1 # return lags that will include in further ar
    print('order', order)
    solution_coeffs= calc_a(endog, order)
    # a, resid = solution_coeffs[0], solution_coeffs[1]
    # print(resid)
    # print('a', a)
    # for i in range(forecast):
    #     endog = np.append(endog, np.dot(a[1:],endog[:-order-1:-1])+a[0])
    #
    # return endog[n:]


x = np.arange(7,dtype=float)
x[0] = 0.1
x[3] = 100
print('x', x)
print('forecast',ar(x,3))
#endog = np.array([1,4,9,16,25]) # ok
#endog = np.array([1,1.41,3**0.5,2,5**0.5,6**0.5])
# step = 7
# endog_forecast = arma(endog, step)
#
#
# def compare_vals(name, real, predicted, reconstructed=None):
#         fig = plt.figure()
#         axes = plt.axes()
#         r = np.arange(len(real))
#         axes.set_title(name)
#         axes.set_xlim(0, len(real))
#         axes.grid()
#         axes.plot(r, predicted, label='predicted')
#         if reconstructed != None:
#             axes.plot(r, reconstructed, label='reconstructed')
#         axes.plot(r, real, label='real')
#         axes.legend(loc='upper right', fontsize=16)
#         fig.show()
#
#
# #compare_vals('some', endog, np.append(endog, endog_forecast))
# fig = plt.figure()
# axes = plt.axes()
# data = np.append(endog, endog_forecast)
# r = np.arange(len(data))
# axes.set_xlim(0, len(data))
# axes.grid()
# axes.plot(r,data)
# plt.show()

