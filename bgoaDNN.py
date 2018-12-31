from random import randint
from numpy import genfromtxt
from sklearn import preprocessing
import math,numpy
from deepNN import DeepNeuralNetwork


def adaptive_haar_wavelet(binary,k_value):

    min_max_scaler = preprocessing.MinMaxScaler()
    binary = min_max_scaler.fit_transform(binary)
    binary=binary.tolist()
    for i in range(len(binary)):
        m=k_value[i]
        m=(2**m)
        k=randint(0,m//2)
        for j in range(len(binary[i])):
            x=binary[i][j]
            if x >= (k/float(m)) and x <= ((k+0.5)/float(m)):
                binary[i][j]=int(1)
            else:
                binary[i][j]=int(0)

    return(binary)
        
def commonality_based_crossover(x):
    y=[]
    x=numpy.array(x)
    l=x.shape[1]
    flag=numpy.all(x==x[0,:],axis=0)
    x=x.T
    for i in range(l):
        if flag[i]!=True:
            y.append(i)
    z=(x[y,:].T)
    z=numpy.reshape(z,[1,z.shape[0]*z.shape[1]])
    z=numpy.reshape(z,x[y,:].shape)
    xx=[0*n for n in range(l)]
    count=0
    for i in range(l):
        if flag[i]!=True:
            xx[i]=list(z[count])
            count+=1
        else:
            xx[i]=list(x[i])
    xx=numpy.array(xx).T
    return(list(xx))

def goa_haar_dffnn(no_of_generation,no_of_population,bitgenerated,feature_size,X,Y):
    
    binary_bits=[[]*n for n in range(no_of_population)]
    cMax=1
    cMin=0.00001
    best_fitness=-100.00
    best_accuracy=-100.00
    fitness_array=[]
    best_fitness_monitor=[]

    for j in range(no_of_population):
        for k in range(bitgenerated):
            x=randint(0,1)
            binary_bits[j].append(x)

    print("Generation0:")
    for index in range(no_of_population):

        list_with_one=[]
        
        for t in range(feature_size):
            if binary_bits[index][t]==1:
                list_with_one.append(t)
        if len(list_with_one)<=5:
            list_with_one=(range(20,35))
        accuracy=DeepNeuralNetwork(list_with_one,binary_bits[index][feature_size:-4],X[:,list_with_one],Y)
        
        fitnessA=(0.9099*accuracy)
        fitnessB=0.0001*(1-(len(list_with_one)/float(feature_size)))

        fitness=fitnessA+fitnessB
        fitness_array.append(fitness)
        
        if fitness > best_fitness:
            best_fitness=fitness
            best_fitness_index=index
            best_accuracy=accuracy

    print("Best fitness: "+str(best_fitness)+" and "+"Best accuracy: "+str(best_accuracy))
    best_fitness_monitor.append(best_fitness)


    for index in range(no_of_generation-1):

        print("Generation"+str(index+1)+':')

        c=cMax-(index+1)*((cMax-cMin)/float(no_of_generation))
        k_value=[]

        for pop in range(no_of_population):    
            
            k_instant=[str(ith) for ith in binary_bits[pop][-4:-1]]
            k_instant=''.join(k_instant)
            k_value.append(int(k_instant,2))
            for dec in range(bitgenerated):
            
                summation=0
                current_bit=binary_bits[pop][dec]

                for k in range(no_of_population):
                
                    if k!=pop:
                        a=abs(current_bit-binary_bits[k][dec])
                        b=current_bit-binary_bits[k][dec]
                        a=round(a,3)
                        summation+=0.5*c*math.exp(a/1.5)*0.5-math.exp(-a)*b*3*current_bit
                
                binary_bits[pop][dec]=c*summation+binary_bits[best_fitness_index][dec]

        binary_bits=adaptive_haar_wavelet(binary_bits,k_value)

        for t in range(no_of_population):

            list_with_one=[]
            
            for s in range(feature_size):
                if binary_bits[t][s]==1:
                    list_with_one.append(s)
            if len(list_with_one)<=5:
                list_with_one=(range(20,35))
            accuracy=DeepNeuralNetwork(list_with_one,binary_bits[t][feature_size:-4],X[:,list_with_one],Y)
            fitnessA=(0.9099*accuracy)
            fitnessB=0.0001*(1-(len(list_with_one)/float(feature_size)))
            fitness=fitnessA+fitnessB
            fitness_array.append(fitness)
        
            if fitness > best_fitness:
                best_fitness=fitness
                best_fitness_index=index
                best_accuracy=accuracy

        print("Best fitness: "+str(best_fitness)+" and "+"Best accuracy: "+str(best_accuracy))
        best_fitness_monitor.append(best_fitness)

        if len(best_fitness_monitor)>=4 and best_fitness_monitor[-1]==best_fitness_monitor[-2]==best_fitness_monitor[-3]:
            print("Maximum fitness value is repeated for three times, Initiating Crossover")
            binary_bits=commonality_based_crossover(binary_bits)

no_of_generation=3
no_of_population=2

max_lr_len=3
min_lr_len=3
momentum_len=4
drop_out=6
haar_para_len=4

feature_size=55

bitgenerated=max_lr_len+min_lr_len+momentum_len+drop_out+haar_para_len+feature_size

f1= genfromtxt('FocX.csv', delimiter=',', skip_header=0, usecols=(range(0,feature_size)))
f2= genfromtxt('FocY.csv', delimiter=',', skip_header=0, usecols=(range(0,feature_size)))
f3= genfromtxt('FocXY.csv', delimiter=',', skip_header=0, usecols=(range(0,feature_size)))
f4= genfromtxt('NfocX.csv', delimiter=',', skip_header=0, usecols=(range(0,feature_size)))
f5= genfromtxt('NfocXY.csv', delimiter=',', skip_header=0, usecols=(range(0,feature_size)))
f6= genfromtxt('NfocY.csv', delimiter=',', skip_header=0, usecols=(range(0,feature_size)))

#focal=list(f1)+list(f2)+list(f3)
#non_focal=list(f4)+list(f5)+list(f6)
focal=list(f3)
non_focal=list(f5)

y_focal=[0]*len(focal)
y_nfocal=[1]*len(non_focal)

X=numpy.array(focal+non_focal)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
Y=numpy.array(y_focal+y_nfocal)

goa_haar_dffnn(no_of_generation,no_of_population,bitgenerated,feature_size,X,Y)



        