#importing necessary libraries
from keras.models import Sequential
from keras.layers import Dense,Dropout,Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import Callback,EarlyStopping
import keras.backend as K
from keras import regularizers
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy

#value for hyperparameters for Learning rate Annealing
max_lr_value=1e-2
min_lr_value=1e-5
momentum_value=0.99
#Import data from the .CSV file
f1= genfromtxt('Focal_Db10_features.csv', delimiter=',', skip_header=0)
f2= genfromtxt('NFocal_Db10_features.csv', delimiter=',', skip_header=0)

#make them as list and add them to make dataset X
focal=list(f1)
non_focal=list(f2)
X=numpy.array(focal+non_focal)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
print(X.shape)

#add class label for them
y_focal=[0]*len(focal)
y_nfocal=[1]*len(non_focal)
Y=numpy.array(y_focal+y_nfocal)

#learning rate annealing callback
trigger_flag=0
threshold=0.1

class OptimizerChanger(Callback):

    def on_epoch_end(self,epoch,logs={}):
        if epoch==100:            
            print("Changing to learning rate annelaing with SGD")
            sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            epoch_size=200;batch_size=2048        
            #callback initialization for learning rate annealing 
            schedule =SGDRScheduler(min_lr=min_lr_value,max_lr=max_lr_value,steps_per_epoch=numpy.ceil(epoch_size/batch_size),lr_decay=0.70,cycle_length=10,mult_factor=1.5)    
            history=model.fit(X_train, Y_train,validation_data=(X_valid,Y_valid),epochs=300,batch_size=2048,callbacks=[schedule],verbose=0,shuffle=True)

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
        scores = model.evaluate(X[test], Y[test], verbose=1)
        cv.append(scores[1])
        print(scores[1])
        


#k fold validation data split
skfold = StratifiedKFold(n_splits=10,shuffle=True)
cv=[]

for (train,test) in skfold.split(X,Y):        
    #data split for training and validation 
    X_train, X_valid, Y_train, Y_valid = train_test_split(X[train],Y[train], test_size=0.2, shuffle= True,stratify=Y[train])
    #define model
    model = Sequential()
    #input layer 
    model.add(Dense(100, input_dim=X.shape[1], activation='relu',kernel_regularizer=regularizers.l1(0.01)))
    #Hidden layer 1
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    #Hidden layer 2
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    #hidden layer 3
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    #Hidden layer 4
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    #Hidden layer 5
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    #Hidden layer 6
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    #Output layer
    model.add(Dense(1, activation='sigmoid'))
    #compile the model
    stopper=OptimizerChanger()
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    history=model.fit(X_train, Y_train,validation_data=(X_valid,Y_valid),epochs=101,batch_size=2048,callbacks=[stopper],verbose=0,shuffle=True)
    #define optimizing algorithm setting
    #model fitting for the above settings
    #Evaluate the trained model on unseen test dataset for checking generalization
    
print(cv)
sss=sum(cv)/float(len(cv))
print(sss)
