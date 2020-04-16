from google.colab import drive
from shutil import copyfile
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from keras.utils.np_utils import to_categorical 
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


def load_data(labels=['up', 'three', 'four', 'stop', 'left', 'on', 'six', 'right', 'go',
                    'seven', 'no', 'one', 'off', 'yes', 'nine', 'zero', 'two', 'down', 'five', 'eight'],
                SAMPLES = 8000, TEST_SPLIT = 0.2, VAL_SPLIT = 0.2):

    drive.mount('/content/drive')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    new_dir = "speech_data"
    path = os.path.join(dir_path, new_dir) 
    
    DATA_PATH = path+'/'
    
    os.mkdir(path)
    
    new_dir = "model_weights"
    path = os.path.join(path, new_dir) 
    
    os.mkdir(path)


    for label in labels:
        copyfile('/content/drive/My Drive/cs419_data/'+label+'.mat','/content/speech_data/'+label+'.mat')

    x = np.arange(SAMPLES)
    y = []
    for label in labels:
        print(label)
        m = sio.loadmat(DATA_PATH + label+'.mat')
        m = m['samples']
        x = np.vstack((x,m))
        y = y + [label]*m.shape[0]
        #print(m.shape[0])
        #print("----")

    #Delete the initial row
    x = np.delete(x,0,axis = 0)

    y_array = np.array(y).reshape(-1,1)

    ohe = OneHotEncoder()
    ohe.fit(y_array)
    y_enc = ohe.transform(y_array).toarray()

    x1,x_test,y1,y_test = train_test_split(x,y_enc,test_size=TEST_SPLIT,random_state=12,shuffle = True)
    x_train,x_val,y_train,y_val = train_test_split(x1,y1,test_size=(VAL_SPLIT/(1-TEST_SPLIT)),random_state=12,shuffle = True)

    print("Test Set X shape %s" % (x_test.shape,))
    print("Test Set Y shape %s" % (y_test.shape,))
    
    print("Val Set X shape %s" % (x_val.shape,))
    print("Val Set Y shape %s" % (y_val.shape,))

    print("Train Set X shape %s" % (x_train.shape,))
    print("Train Set Y shape %s" % (y_train.shape,))
    
    return x_train,y_train,x_val,y_val,x_test,y_test, labels, ohe, DATA_PATH


def plot_graph(fit_hist, size = (15,8)):


    plt.figure(1, figsize = size) 
        
    plt.subplot(221)  
    plt.plot(fit_hist.history['acc'])  
    plt.plot(fit_hist.history['val_acc'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'valid']) 
    plt.grid(True)

    plt.subplot(222)  
    plt.plot(fit_hist.history['loss'])  
    plt.plot(fit_hist.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'valid']) 
    plt.grid(True)

    plt.show()


def plot_confusion_matrix(y_pred,y_test,labels, ohe,size=(10,7)):

    N = len(labels)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    df_cm = pd.DataFrame(matrix, index = [ohe.inverse_transform(to_categorical(i, num_classes= N).reshape(1,-1))[0,0] for i in range(N)],
                    columns = [ohe.inverse_transform(to_categorical(i, num_classes=N).reshape(1,-1))[0,0] for i in range(N)])
    plt.figure(figsize = size)
    sn.heatmap(df_cm, annot=True,cmap='GnBu')


def model_maker(x_train,y_train,x_val,y_val,layers_info, DATAPATH, NUM_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE):
    K.clear_session()

    model = Sequential()
    
    for layer_ in layers_info:
        print("__--__")
        print(layer_[0])
        print(layer_[1])
        model.add(layer_[0](**layer_[1]))

    adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer= adm , metrics=['acc'])
    model.summary()

    name = DATAPATH + "model_weights/LR-weights.{epoch:02d}-{val_loss:.3f}.hdf5"
    early_stopper = EarlyStopping(monitor = 'val_acc', patience = EARLY_STOP_PATIENCE)
    checkpointer = ModelCheckpoint(filepath = name, monitor = 'val_acc', save_best_only = True, mode = 'auto')

    cbk = [checkpointer, early_stopper]

    fit_hist = model.fit(x = x_train, 
          y = y_train,
          batch_size = BATCH_SIZE,
          epochs = NUM_EPOCHS,
          verbose = 1,
          validation_data=(x_val,y_val),
          callbacks=cbk)

    return model,fit_hist
