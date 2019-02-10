import pandas,numpy as np  
from pyentrp import entropy

def gaussmf(x, mean, sigma):
    """
    Gaussian fuzzy membership function.
    Parameters
    ----------
    x : 1d array or iterable
        Independent variable.
    mean : float
        Gaussian parameter for center (mean) value.
    sigma : float
        Gaussian parameter for standard deviation.
    Returns
    -------
    y : 1d array
        Gaussian membership function for x.
    """
    return np.exp(-((x - mean)**2.) / (2 * sigma**2.))

def fuzzyent(d1):
    sa1=[]
    #d1=np.rint(d1)
    print("Fuzzy started")
    for i in range(d1.shape[0]):
        print(i,end=" ",flush=True)
        X=d1[i]
        X=gaussmf(X,0,1);X=np.array(X)
        X=np.rint(X)
        ee=entropy.shannon_entropy(list(X))
        sa1.append(ee)
    print("Fuzzy Finished")
    return(sa1)

print("==D4 loading==")
d4= pandas.read_csv('Db10Coeffecients/Db10NFZD4.csv', sep=',', header=None)
d4=np.array(d4)
sa4=fuzzyent(d4)
del(d4)

print("==D5 loading==")
d5= pandas.read_csv('Db10Coeffecients/Db10NFZA4.csv', sep=',', header=None)
d5=np.array(d5)
sa5=fuzzyent(d5)
del(d5)

print("==D3 loading==")
d3= pandas.read_csv('Db10Coeffecients/Db10NFZD3.csv', sep=',', header=None)
d3=np.array(d3)
sa3=fuzzyent(d3)
del(d3)

print("==D2 loading==")
d2= pandas.read_csv('Db10Coeffecients/Db10NFZD2.csv', sep=',', header=None)
d2=np.array(d2)
sa2=fuzzyent(d2)
del(d2)

print("==D1 loading==")
d1= pandas.read_csv('Db10Coeffecients/Db10NFZD1.csv', sep=',', header=None)
d1=np.array(d1)
sa1=fuzzyent(d1)
del(d1)

print("Making array")
X=[sa1,sa2,sa3,sa4,sa5]
X=np.array(X)
X=X.T
a = np.asarray(X)
print("Writing")
print(X.shape)
np.savetxt("NFocal_Db10_fuzzy.csv", a, delimiter=",")
print("Finished successfully")
