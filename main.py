import warnings
warnings.simplefilter("ignore", UserWarning)

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
#from keras.callbacks import EarlyStopping,TensorBoard
import numpy,math
from random import randint
from numpy import genfromtxt
from sklearn import preprocessing
from keras.callbacks import Callback
import keras.backend as K


class SGDRScheduler(Callback):

    def __init__(self,min_lr,max_lr,steps_per_epoch,lr_decay=1,cycle_length=10,mult_factor=2):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.batch_since_restart = 0
        self.next_restart = cycle_length
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + numpy.cos(fraction_to_restart * numpy.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = numpy.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)


def param_calc(bits):
    
    print(bits)

    Dropout_rate=bits[0:6]
    momentum_value=bits[6:10]
    max_lr_value=bits[10:13]
    min_lr_value=bits[13:16]
    
    Dropout_rate=[str(int(i)) for i in Dropout_rate]
    Dropout_rate=''.join(Dropout_rate)
    Dropout_rate=int(Dropout_rate,2)
    if Dropout_rate==0:
        Dropout_rate=5

    Dropout_rate=Dropout_rate/float(100)

    momentum_value=[str(int(i)) for i in momentum_value]
    momentum_value=''.join(momentum_value)
    momentum_value=int(momentum_value,2)
    if momentum_value==0:
        momentum_value=5

    momentum_value=momentum_value/float(100)
    momentum_value=0.85+momentum_value

    max_lr_value=[str(int(i)) for i in max_lr_value]
    max_lr_value=''.join(max_lr_value)
    max_lr_value=int(max_lr_value,2)
    if max_lr_value==0:
        max_lr_value=1

    max_lr_value=max_lr_value*(10**-2)

    min_lr_value=[str(int(i)) for i in min_lr_value]
    min_lr_value=''.join(min_lr_value)
    min_lr_value=int(min_lr_value,2)
    if min_lr_value==0:
        min_lr_value=1

    min_lr_value=min_lr_value*(10**-5)
    
    return(Dropout_rate,momentum_value,max_lr_value,min_lr_value)
    
def DeepNeuralNetwork(list_with_one,bits,X,Y):
    
    skfold = StratifiedKFold(n_splits=10,shuffle=True)

    Dropout_rate,momentum_value,max_lr_value,min_lr_value = param_calc(bits)
    cvscore=[]
    
    for (train, test) in skfold.split(X,Y):
    
        X_train, X_valid, Y_train, Y_valid = train_test_split(X[train],Y[train], test_size=0.2, shuffle= True,stratify=Y[train])
        model = Sequential()
        model.add(Dense(80, input_dim=len(list_with_one), activation='relu'))
        model.add(Dropout(Dropout_rate))
        model.add(Dense(90, activation='relu'))
        model.add(Dropout(Dropout_rate))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(Dropout_rate))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        sgd = SGD(lr=0.1, momentum=momentum_value, decay=0.0, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #early_stopping_monitor = [EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')]
        #tensorboard = [TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True, write_images=True)]
        epoch_size=1000;batch_size=2250
        schedule = SGDRScheduler(min_lr=min_lr_value,max_lr=max_lr_value,steps_per_epoch=numpy.ceil(epoch_size/batch_size),lr_decay=0.9,cycle_length=10,mult_factor=1.5)
        #history=model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid), epochs=500,callbacks=[schedule,early_stopping_monitor,tensorboard],verbose=2,shuffle=True)
        history=model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid), epochs=10,batch_size=1250,callbacks=[schedule],verbose=0,shuffle=True)
        scores = model.evaluate(X[test], Y[test], verbose=0)
        cvscore.append(scores[1]*100.000)
        #plot_graph(history)

    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscore), numpy.std(cvscore)))
    return(numpy.mean(cvscore))

def plot_graph(history):

    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy_plot.png')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss_plot.png')

def adaptive_haar_wavelet(binary,k_value):

    min_max_scaler = preprocessing.MinMaxScaler()
    binary = min_max_scaler.fit_transform(binary)
    binary=binary.tolist()
    for i in range(len(binary)):
        m=k_value[i]
        m=(2**m)
        if m<=1:
            m=2
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
        print(accuracy)
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
            print(accuracy)
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

no_of_generation=10
no_of_population=3

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