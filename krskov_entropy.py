import numpy,pandas
from scipy.special import gamma,psi
from scipy.linalg import det
from numpy import pi
from sklearn.neighbors import NearestNeighbors

def kraskov_entropy(d1):
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
        return (d*numpy.mean(numpy.log(r + numpy.finfo(X.dtype).eps))+ numpy.log(volume_unit_ball) + psi(n) - psi(k))
    kd1=[]
    for i in range(d1.shape[0]):
        print(i,end=" ",flush=True)
        x=d1[i]
        x=numpy.array(x).reshape(-1,1)
        kd1.append(entropy(x, k))
    return(kd1)

d1= pandas.read_csv('Db10Coeffecients/Db10NFZD1.csv', sep=',', header=None)
d1=numpy.array(d1)
k1=kraskov_entropy(d1)

d2= pandas.read_csv('Db10Coeffecients/Db10NFZD2.csv', sep=',', header=None)
d2=numpy.array(d2)
k2=kraskov_entropy(d2)

d3= pandas.read_csv('Db10Coeffecients/Db10NFZD3.csv', sep=',', header=None)
d3=numpy.array(d3)
k3=kraskov_entropy(d3)

d4= pandas.read_csv('Db10Coeffecients/Db10NFZD4.csv', sep=',', header=None)
d4=numpy.array(d4)
k4=kraskov_entropy(d4)

d5= pandas.read_csv('Db10Coeffecients/Db10NFZA4.csv', sep=',', header=None)
d5=numpy.array(d5)
k5=kraskov_entropy(d5)

X=[k1,k2,k3,k4,k5]
X=numpy.array(X)
X=X.T
a = numpy.asarray(X)
print("Writing")
print(X.shape)
numpy.savetxt("Kraskov_NFocal_Db10_features.csv", a, delimiter=",")
print("Finished successfully")
