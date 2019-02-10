""" BEFORE EXTRACTING FEATURES FROM WAVELET COEFFECIENTS CHECK THE SHAPE OF INPUT DATASET!!!"""
import pandas,math,time
from pyentrp import entropy
import numpy as np
from scipy.special import gamma,psi
from scipy.linalg import det
from numpy import pi
from sklearn.neighbors import NearestNeighbors

def kraskov_entropy(d1):
    print("Kraskov Started")
    k=4
    def nearest_distances(X, k):
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        d, _ = knn.kneighbors(X)
        return d[:, -1]
    def entropy(X, k):
        r = nearest_distances(X, k)
        n, d = X.shape
        volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
        return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))+ np.log(volume_unit_ball) + psi(n) - psi(k))
    kd1=[]
    for i in range(d1.shape[0]):
        print(i,end=" ",flush=True)
        x=d1[i]
        x=np.array(x).reshape(-1,1)
        kd1.append(entropy(x, k))
    print("Kraskov Finished")
    return(kd1)

def renyi_entropy(d1):
    print("Renyi started")
    d1=np.rint(d1)
    rend1=[]
    alpha=2    
    for i in range(d1.shape[0]):
        X=d1[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Renyi_En=(1/float(1-alpha))*(math.log(summation,2))
        rend1.append(Renyi_En)
    print("Renyi Finished")
    return(rend1)

def permu(d1):
    pd1=[]
    print("Permutation started")
    for i in range(d1.shape[0]):
        X=d1[i]
        pd1.append(entropy.permutation_entropy(X,3,1))
    print("Permutation Finished")
    return(pd1)
    
def sampl(d1):
    sa1=[]
    print("Sample started")
    for i in range(d1.shape[0]):
        X=d1[i]
        std_X = np.std(X)
        ee=entropy.sample_entropy(X,2,0.2*std_X)
        sa1.append(ee[0])
    print("Sample Finished")
    return(sa1)

def shan(d1):
    sh1=[]
    print("Shannon started")
    d1=np.rint(d1)
    for i in range(d1.shape[0]):
        X=d1[i]
        sh1.append(entropy.shannon_entropy(X))
    print("Shannon Finished")
    return(sh1)

print("==D4 loading==")
d4= pandas.read_csv('Db10NFZD4.csv', sep=',', header=None)
d4=np.array(d4)
sa4=sampl(d4)
r4=renyi_entropy(d4)
p4=permu(d4)
sh4=shan(d4)
ka4=kraskov_entropy(d4)
del(d4)

print("==D5 loading==")
d5= pandas.read_csv('Db10NFZA4.csv', sep=',', header=None)
d5=np.array(d5)
sa5=sampl(d5)
r5=renyi_entropy(d5)
p5=permu(d5)
sh5=shan(d5)
ka5=kraskov_entropy(d5)
del(d5)

print("==D3 loading==")
d3= pandas.read_csv('Db10NFZD3.csv', sep=',', header=None)
d3=np.array(d3)
sa3=sampl(d3)
r3=renyi_entropy(d3)
p3=permu(d3)
sh3=shan(d3)
ka3=kraskov_entropy(d3)
del(d3)

print("==D2 loading==")
d2= pandas.read_csv('Db10NFZD2.csv', sep=',', header=None)
d2=np.array(d2)
print(d2.shape)
sa2=sampl(d2)
r2=renyi_entropy(d2)
p2=permu(d2)
sh2=shan(d2)
ka2=kraskov_entropy(d2)
del(d2)

print("==D1 loading==")
d1= pandas.read_csv('Db10NFZD1.csv', sep=',', header=None)
d1=np.array(d1)
print(d1.shape)
sa1=sampl(d1)
r1=renyi_entropy(d1)
p1=permu(d1)
sh1=shan(d1)
ka1=kraskov_entropy(d1)
del(d1)

print("Making array")
X=[r1,r2,r3,r4,r5,p1,p2,p3,p4,p5,sh1,sh2,sh3,sh4,sh5,sa1,sa2,sa3,sa4,sa5,ka1,ka2,ka3,ka4,ka5]
X=np.array(X)
X=X.T
a = np.asarray(X)
print("Writing")
print(X.shape)
np.savetxt("NFocalZ_Db10_features.csv", a, delimiter=",")
print("Finished successfully")
