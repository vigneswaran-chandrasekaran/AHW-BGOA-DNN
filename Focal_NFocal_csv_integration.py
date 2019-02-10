import numpy,glob,errno,pandas,pywt

path="Data_Focal_Full/*.txt"
files=glob.glob(path)
files=sorted(files)
db = pywt.Wavelet('db12')
a=[];d4=[];d3=[];d2=[];d1=[]
for i in range(len(files)):
    
    print(i,end=' ',flush=True)
    X=pandas.read_csv(files[i],sep=",",header=None)
    X[2]=X[0]-X[1]

    #x.append(X[0].values)
    #y=X[1].values
    z=X[2].values

    cA,cD4,cD3,cD2,cD1= pywt.wavedec(z,db,level=4)
    
    a.append(cA)
    d4.append(cD4)
    d3.append(cD3)
    d2.append(cD2)
    d1.append(cD1)

numpy.savetxt("Db12FZD1.csv", d1, delimiter=",")
del(d1)
numpy.savetxt("Db12FZD2.csv", d2, delimiter=",")
del(d2)
numpy.savetxt("Db12FZD3.csv", d3, delimiter=",")
del(d3)
numpy.savetxt("Db12FZD4.csv", d4, delimiter=",")
del(d4)
numpy.savetxt("Db12FZA4.csv", a, delimiter=",")
del(a)

############# Non Focal###########################
path="Data_NFocal_Full/*.txt"
files=glob.glob(path)
files=sorted(files)
db = pywt.Wavelet('db12')
a=[];d4=[];d3=[];d2=[];d1=[]
for i in range(len(files)):
    
    print(i,end=' ',flush=True)
    X=pandas.read_csv(files[i],sep=",",header=None)
    X[2]=X[0]-X[1]

    #x.append(X[0].values)
    #y=X[1].values
    z=X[2].values

    cA,cD4,cD3,cD2,cD1= pywt.wavedec(z,db,level=4)
    
    a.append(cA)
    d4.append(cD4)
    d3.append(cD3)
    d2.append(cD2)
    d1.append(cD1)

numpy.savetxt("Db12NFZD1.csv", d1, delimiter=",")
del(d1)
numpy.savetxt("Db12NFZD2.csv", d2, delimiter=",")
del(d2)
numpy.savetxt("Db12NFZD3.csv", d3, delimiter=",")
del(d3)
numpy.savetxt("Db12NFZD4.csv", d4, delimiter=",")
del(d4)
numpy.savetxt("Db12NFZA4.csv", a, delimiter=",")
del(a)