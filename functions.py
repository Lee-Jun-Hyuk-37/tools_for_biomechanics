import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

# NoLiTSA 라이브러리 import

!git clone https://github.com/manu-mannattil/nolitsa.git
!pip install ./nolitsa

import nolitsa

from nolitsa import data, delay, dimension, lyapunov
from sklearn.linear_model import LinearRegression


"""
데이터를 기준점으로 자르고 resampling 하는 함수

input으로 raw_data와 heel_strike 지점 그리고 원하는 resample_dimension을 넣으면 정리된 값이 나온다. 

예시)
data.shape => (36123,)
points.shape => (530,)
resample_dim = 100
flatten=False

그러면 output.shape => (530, 100) 이 나온다. 

flatten=True 로 하면 output.shape => (530*100,) 이 나온다. 
"""

def cut_data_by_cycle_and_resample(data, points, resample_dim, flatten=False):
    resampled_data = np.zeros((len(points)-1, resample_dim))

    for i in range(1, len(points)):
        tmp = np.array(data[points[i-1]:points[i]])
        resampled_tmp = np.zeros(((len(tmp)-1)*resample_dim))
        
        for j in range((len(tmp)-1)*resample_dim):
            if j % resample_dim == 0:
                resampled_tmp[j] = tmp[j//resample_dim]
            else:
                resampled_tmp[j] = np.nan

        resampled_tmp = pd.Series(resampled_tmp)
        resampled_tmp.iloc[-1] = tmp[-1]

        resampled_tmp = resampled_tmp.interpolate(method='cubic')
        resampled_tmp = np.array(resampled_tmp)

        for j in range(len(resampled_tmp)-1, -1, -1):
            if j % (len(tmp)-1) != 0:
                resampled_tmp = np.delete(resampled_tmp, j, axis=0)
        
        resampled_data[i-1, :] = resampled_tmp
    
    if flatten == False:
        return resampled_data
    else:
        return resampled_data.reshape(-1)
        

"""
input으로 raw_data와 heel_strike 지점, 원하는 resample_dimension과 뽑고 싶은 값을 넣으면 미분된 값이 resample 되어서 나온다. 

예시)
data.shape => (36123,)
points.shape => (530,)
resample_dim = 100
flatten=False

그러면 output.shape => (530, 100) 이 나온다. 

order의 
0번째 index는 도함수
1번째 index는 이계도함수
2번째 index는 삼계도함수
를 의미한다. 

셋 모두를 뽑고 싶으면 order=[1, 1, 1]
도함수와 삼계도함수만 뽑고 싶으면 order=[1, 0, 1]
이런식으로 입력하면 된다.
"""

def differentiate_data_and_resample(data, points, sampling_rate, resample_dim, order = [1, 1, 1]):
    if order[0] == 1:
        resampled_first_derivative = np.zeros((len(points)-1, resample_dim))
    if order[1] == 1:
        resampled_second_derivative = np.zeros((len(points)-1, resample_dim))
    if order[2] == 1:
        resampled_third_derivative = np.zeros((len(points)-1, resample_dim))

    for i in range(len(points)-1):
        tmp = np.array(data[points[i]:points[i+1]])
        first = (tmp[1:] - tmp[:-1])/sampling_rate
        second = (first[1:] - first[:-1])/sampling_rate
        third = (second[1:] - second[:-1])/sampling_rate
        if order[0] == 1:
            resampled_first_derivative[i] = cut_data_by_cycle_and_resample(first, [0, len(first)], 100, flatten=True)
        if order[1] == 1:
            resampled_second_derivative[i] = cut_data_by_cycle_and_resample(second, [0, len(second)], 100, flatten=True)
        if order[2] == 1:
            resampled_third_derivative[i] = cut_data_by_cycle_and_resample(third, [0, len(third)], 100, flatten=True)

    result = []
    if order[0] == 1:
        result.append(resampled_first_derivative)
    if order[1] == 1:
        result.append(resampled_second_derivative)
    if order[2] == 1:
        result.append(resampled_third_derivative)

    return result


def Lyapunov_Exponet_full_process(data, cycle_dim):
    mi = delay.dmi(data, maxtau = cycle_dim*3, bins=int((len(data)/5)**0.5))

    plt.figure()
    plt.plot(mi)
    plt.title('Delayed mutual information')
    plt.xlabel('time delay')
    plt.ylabel('Mutual Information')
    plt.show()

    for i in range(100):
        if mi[i-1] > mi[i] and mi[i] < mi[i+1]:
            tau = i
            print("first minimum of AMI:", tau)
            break
    
    dim = np.arange(1, 15+1)
    f1, f2, f3 = dimension.fnn(data, tau=tau, dim=dim, metric='euclidean')
    plt.figure()
    plt.xlabel(r'Embedding dimension $d$')
    plt.ylabel(r'FNN (%)')
    plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
    plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
    plt.plot(dim, 100 * f3, 'rs-', label=r'Test I or II')
    plt.legend()
    plt.show()

    for i in range(1, 15+1):
        if f3[i-1] == 0:
            embedding_dim = i
            print("embedding dimension:", embedding_dim)
            break
    else:
        print("Error: GFNN doesn't converge to 0 until dimension 15")
        return 'Error', 'Error'

    d = lyapunov.mle_embed(data, dim=[embedding_dim], tau=tau, maxt=cycle_dim*10)
    t = np.arange(cycle_dim*10).reshape(-1, 1)

    short_term = LinearRegression()
    short_term.fit(t[:cycle_dim]/cycle_dim, d[0, :cycle_dim])
    short_term_LE = short_term.coef_

    long_term = LinearRegression()
    long_term.fit(t[cycle_dim*4:]/cycle_dim, d[0, cycle_dim*4:])
    long_term_LE = long_term.coef_

    plt.figure()
    plt.xlabel(r'Steps #')
    plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
    plt.plot(t/cycle_dim, d[0])
    plt.plot(t[:cycle_dim]/cycle_dim, short_term.predict(t[:cycle_dim]/cycle_dim))
    plt.plot(t[cycle_dim*4:]/cycle_dim, long_term.predict(t[cycle_dim*4:]/cycle_dim))

    plt.show()

    print("Short-term Lyapunov Exponent: {}".format(short_term_LE[0]))
    print("Long-term Lyapunov Exponent: {}".format(long_term_LE[0]))

    return short_term_LE, long_term_LE


def reconstruct_multidim(x, dim=1, tau=1):
    for i, tar in enumerate(x):
        if i == 0:
            y = utils.reconstruct(tar, dim, tau)
        else:
            y = np.concatenate([y, utils.reconstruct(tar, dim, tau)], axis=1)
    return y


def _fnn_multidim(d, x, tau=1, R=17.0, A=2.0, metric='euclidean', window=10, maxnum=None):
    y1 = reconstruct_multidim(x[:, :-tau], d, tau)
    y2 = reconstruct_multidim(x, d + 1, tau)

    index, dist = utils.neighbors(y1, metric=metric, window=window,
                                  maxnum=maxnum)

    f1 = np.abs(y2[:, -1] - y2[index, -1]) / dist > R
    f2 = utils.dist(y2, y2[index], metric=metric) / np.std(x) > A
    f3 = f1 | f2

    return np.mean(f1), np.mean(f2), np.mean(f3)


def fnn_multidim(x, dim=[1], tau=1, R=17.0, A=2.0, metric='euclidean', window=10, maxnum=None, parallel=True):
    if parallel:
        processes = None
    else:
        processes = 1

    return utils.parallel_map(_fnn_multidim, dim, (x,), {
                              'tau': tau,
                              'R': R,
                              'A': A,
                              'metric': metric,
                              'window': window,
                              'maxnum': maxnum
                              }, processes).T


def mle(y, maxt=500, window=10, metric='euclidean', maxnum=None):
    index, dist = utils.neighbors(y, metric=metric, window=window, maxnum=maxnum)
    m = len(y)
    maxt = min(m - window - 1, maxt)

    d = np.empty(maxt)
    d[0] = np.mean(np.log(dist))

    for t in range(1, maxt):
        t1 = np.arange(t, m)
        t2 = index[:-t] + t

        # Sometimes the nearest point would be farther than (m - maxt)
        # in time.  Such trajectories needs to be omitted.
        valid = t2 < m
        t1, t2 = t1[valid], t2[valid]

        d[t] = np.mean(np.log(utils.dist(y[t1], y[t2], metric=metric)))

    return d


def mle_embed_multidim(x, dim=[1], tau=1, window=10, maxt=500, metric='euclidean', maxnum=None, parallel=True):
    if parallel:
        processes = None
    else:
        processes = 1

    yy = [reconstruct_multidim(x, dim=d, tau=tau) for d in dim]

    return utils.parallel_map(mle, yy, kwargs={
                              'maxt': maxt,
                              'window': window,
                              'metric': metric,
                              'maxnum': maxnum
                              }, processes=processes)


def Lyapunov_Exponet_full_process_multidim(data, cycle_dim):
    first_minimums = []

    for tar in data:
        mi = delay.dmi(tar, maxtau = cycle_dim*3, bins=int((len(tar)/5)**0.5))

        plt.figure()
        plt.plot(mi)
        plt.title('Delayed mutual information')
        plt.xlabel('time delay')
        plt.ylabel('Mutual Information')
        plt.show()

        for i in range(100):
            if mi[i-1] > mi[i] and mi[i] < mi[i+1]:
                tau = i
                first_minimums.append(tau)
                print("first minimum of AMI:", tau)
                break

    tau = int(sum(first_minimums)/len(first_minimums))
    print("Time delay(tau): ", tau)


    dim = np.arange(1, 15+1)
    f1, f2, f3 = fnn_multidim(data, dim=dim, tau=tau)    

    plt.figure()
    plt.xlabel(r'Embedding dimension $d$')
    plt.ylabel(r'FNN (%)')
    plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
    plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
    plt.plot(dim, 100 * f3, 'rs-', label=r'Test I or II')
    plt.legend()
    plt.show()

    for i in range(1, 15+1):
        if f3[i-1] == 0:
            embedding_dim = i
            print("embedding dimension:", embedding_dim)
            break
    else:
        print("Error: GFNN doesn't converge to 0 until dimension 15")
        return 'Error', 'Error'

        
    d = mle_embed_multidim(data, dim=[embedding_dim], tau=tau, maxt=cycle_dim*10)
    t = np.arange(cycle_dim*10).reshape(-1, 1)

    short_term = LinearRegression()
    short_term.fit(t[:cycle_dim]/cycle_dim, d[0, :cycle_dim])
    short_term_LE = short_term.coef_

    long_term = LinearRegression() 
    long_term.fit(t[cycle_dim*4:]/cycle_dim, d[0, cycle_dim*4:])
    long_term_LE = long_term.coef_

    plt.figure()
    plt.xlabel(r'Steps #')
    plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
    plt.plot(t/cycle_dim, d[0])
    plt.plot(t[:cycle_dim]/cycle_dim, short_term.predict(t[:cycle_dim]/cycle_dim))
    plt.plot(t[cycle_dim*4:]/cycle_dim, long_term.predict(t[cycle_dim*4:]/cycle_dim))

    plt.show()

    print("Short-term Lyapunov Exponent: {}".format(short_term_LE[0]))
    print("Long-term Lyapunov Exponent: {}".format(long_term_LE[0]))

    return short_term_LE, long_term_LE


