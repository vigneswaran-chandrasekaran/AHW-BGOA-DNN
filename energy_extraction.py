""" BEFORE EXTRACTING FEATURES FROM WAVELET COEFFECIENTS CHECK THE SHAPE OF INPUT DATASET!!!"""
import pandas
from pyentrp import entropy
import numpy as np
import math,time

def energy_ext(d1):
    pd1=[]
    print("Energy started")
    for i in range(d1.shape[0]):
        X=np.array(d1[i])
        eng=np.sum(X**2)
        pd1.append(eng)
    print("Energy Finished")
    return(pd1)
    

print("==D4 loading==")
d4= pandas.read_csv('Db10Coeffecients/Db10NFZD4.csv', sep=',', header=None)
d4=np.array(d4)
eng4=energy_ext(d4)
del(d4)

print("==D5 loading==")
d5= pandas.read_csv('Db10Coeffecients/Db10NFZA4.csv', sep=',', header=None)
d5=np.array(d5)
eng5=energy_ext(d5)
del(d5)

print("==D3 loading==")
d3= pandas.read_csv('Db10Coeffecients/Db10NFZD3.csv', sep=',', header=None)
d3=np.array(d3)
eng3=energy_ext(d3)
del(d3)

print("==D2 loading==")
d2= pandas.read_csv('Db10Coeffecients/Db10NFZD2.csv', sep=',', header=None)
d2=np.array(d2)
eng2=energy_ext(d2)
del(d2)

print("==D1 loading==")
d1= pandas.read_csv('Db10Coeffecients/Db10NFZD1.csv', sep=',', header=None)
d1=np.array(d1)
eng1=energy_ext(d1)
del(d1)

print("Making array")
X=[eng1,eng2,eng3,eng4,eng5]

X=np.array(X)
X=X.T

a = np.asarray(X)
print("Writing")
print(X.shape)
np.savetxt("NFocal_Db10_Energy_features.csv", a, delimiter=",")
print("Finished successfully")
