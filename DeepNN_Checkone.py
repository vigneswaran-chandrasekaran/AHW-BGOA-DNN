import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
#from keras.callbacks import EarlyStopping,TensorBoard
from keras.callbacks import Callback
import keras.backend as K
from numpy import genfromtxt
import numpy,os
from sklearn import preprocessing

Dropout_rate=0.1
max_lr_value=1e-1
min_lr_value=1e-5
momentum_value=0.9

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

start=55;target=61

f1= genfromtxt('FocX.csv', delimiter=',', skip_header=0, usecols=([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,56,57,58,59,60]))
f2= genfromtxt('FocY.csv', delimiter=',', skip_header=0, usecols=([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,56,57,58,59,60]))
f3= genfromtxt('FocXY.csv', delimiter=',', skip_header=0, usecols=([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,56,57,58,59,60]))
f4= genfromtxt('NfocX.csv', delimiter=',', skip_header=0, usecols=([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,56,57,58,59,60]))
f5= genfromtxt('NfocXY.csv', delimiter=',', skip_header=0, usecols=([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,56,57,58,59,60]))
f6= genfromtxt('NfocY.csv', delimiter=',', skip_header=0, usecols=([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,56,57,58,59,60]))


#focal=list(f1)+list(f3)
#non_focal=list(f4)+list(f5)   

focal=list(f1)+list(f2)+list(f3)
non_focal=list(f4)+list(f5)+list(f6)   
#focal=list(f3)
#non_focal=list(f5)

y_focal=[0]*len(focal)
y_nfocal=[1]*len(non_focal)

X=numpy.array(focal+non_focal)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
Y=numpy.array(y_focal+y_nfocal)
skfold = StratifiedKFold(n_splits=10,shuffle=True)
cv=[]

for (train,test) in skfold.split(X,Y):
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(X[train],Y[train], test_size=0.2, shuffle= True,stratify=Y[train])
        model = Sequential()
        model.add(Dense(130, input_dim=int(target-start)+15, activation='relu'))
        #model.add(Dropout(Dropout_rate))
        model.add(Dense(170, activation='relu'))
        #model.add(Dropout(Dropout_rate))
        model.add(Dense(250, activation='relu'))
        #model.add(Dropout(Dropout_rate))
        model.add(Dense(270, activation='relu'))
        model.add(Dense(210, activation='relu'))
        model.add(Dense(170, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        sgd = SGD(lr=0.1, momentum=momentum_value, decay=0.0, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        epoch_size=1000;batch_size=2250
        schedule = SGDRScheduler(min_lr=min_lr_value,max_lr=max_lr_value,steps_per_epoch=numpy.ceil(epoch_size/batch_size),lr_decay=0.9,cycle_length=10,mult_factor=1.5)
        history=model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid), epochs=2000,batch_size=3240,verbose=2,shuffle=True)
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        scores = model.evaluate(X[test], Y[test], verbose=1)
        print(scores)
        os.system('spd-say "Vicky your program has finished"')
        exit(1)
        cv.append(scores[1])
print(cv)
sss=sum(cv)/float(len(cv))
print(sss)

