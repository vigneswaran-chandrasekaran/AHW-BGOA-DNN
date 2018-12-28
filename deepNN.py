import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
#from keras.callbacks import EarlyStopping,TensorBoard
import numpy
from lrscheduler import SGDRScheduler

def param_calc(bits):
    
    Dropout_rate=bits[0:6]
    momentum_value=bits[6:10]
    max_lr_value=bits[10:13]
    min_lr_value=bits[13:16]
    
    Dropout_rate=[str(i) for i in Dropout_rate]
    Dropout_rate=''.join(Dropout_rate)
    Dropout_rate=int(Dropout_rate,2)
    if Dropout_rate==0:
        Dropout_rate=5

    Dropout_rate=Dropout_rate/float(100)

    momentum_value=[str(i) for i in momentum_value]
    momentum_value=''.join(momentum_value)
    momentum_value=int(momentum_value,2)
    if momentum_value==0:
        momentum_value=5

    momentum_value=momentum_value/float(100)
    momentum_value=0.85+momentum_value

    max_lr_value=[str(i) for i in max_lr_value]
    max_lr_value=''.join(max_lr_value)
    max_lr_value=int(max_lr_value,2)
    if max_lr_value==0:
        max_lr_value=1

    max_lr_value=max_lr_value*(10**-2)

    min_lr_value=[str(i) for i in min_lr_value]
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
        history=model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid), epochs=10,batch_size=1500,callbacks=[schedule],verbose=2,shuffle=True)
        scores = model.evaluate(X[test], Y[test], verbose=1)
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

