'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from keras.layers.normalization import BatchNomalization
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from sklearn.model_selection import train_test_split
import numpy as np
import numpy

import time
import pandas as pd
import keras
from keras.utils import plot_model
import pandas
import pandas as pd


#Date from INTERLAD

from Setting_Param import *
#Local_Setting_Param


raw_input = pandas.read_csv(open(str(ADDRESS)+str(CSV_NAME_ABBE)))
#raw_input = numpy.loadtxt(open(str(ADDRESS)+str(CSV_NAME_ABBE)), delimiter=",",skiprows=1)

[information_p, component_p, parameter_p]  = np.hsplit(raw_input, [20,82])

component = np.array(component_p)
parameter = np.array(parameter_p)


#[component_a,component]= numpy.vsplit(component_all,[0])
#[parameter_a,parameter]= numpy.vsplit(parameter_all,[0])


#print("parameter_a",parameter_a)
#print("parameter",parameter)
#print("component_a",component_a)
#print("component",component)

TRAIN_DATA_SIZE=DATA_SIZE_ABBE-10
#DATE_SIZE_ABBE =2448

#parameter_train, parameter_test, component_train, component_test = train_test_split(parameter, component, test_size=0.05, random_state=0)

[parameter_train, parameter_test] = numpy.vsplit(parameter, [TRAIN_DATA_SIZE])
[component_train, component_test] = numpy.vsplit(component, [TRAIN_DATA_SIZE])

batch_size = 4000
epochs = 1000000

es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3000, verbose=1, mode='auto')


def get_model(num_layers, layer_size,bn_where,bn_last):
    model =Sequential()
    model.add(Dense(DATA_SIZE_ABBE, input_dim=62, activation='relu'))

    for i in range(num_layers):
        if num_layers != 1:

            if bn_where==0:
                model.add(BatchNormalization(mode=0))
            model.add(Dense(layer_size))
            if bn_where==1:
                model.add(BatchNormalization(mode=0))

            model.add(Activation('relu'))
            if bn_where==2:
                model.add(BatchNormalization(mode=0))

            model.add(BatchNormalization(mode=0))
            if bn_where==3:
                model.add(BatchNormalization(mode=0))

            model.add(Dropout(0.2))
            if bn_where==4:
                model.add(BatchNormalization(mode=0))

    if bn_last ==1:
        model.add(BatchNormalization(mode=0))


    model.add(Dense(1))

    model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    return model


for num_layers in [1,2,3,4,5]:
    for layer_size in[64,128,256,512,1024]:
        for bn_where in [0,1,2,3,4]:
            for bn_last in [0,1]:


                model =get_model(num_layers,layer_size,bn_where,bn_last)

                model.fit(component_train, parameter_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(component_test, parameter_test),
                          callbacks=[es_cb])

                score = model.evaluate(component_test, parameter_test, verbose=0)
                parameter_predict=model.predict(component_test, batch_size=32, verbose=1)

                model.save('C:\Deeplearning/model/ABBE/model_ABBE_' +str(round(score[0],2))
                           +'_numlayer'+ str(num_layers) +'_layersize' + str(layer_size) +'bn_where is '
                           + str(bn_where)+ 'bn_last is '+ str(bn_last)+ '.h5')
                #plot_model(model, to_file='C:\Deeplearning/model.png')
                #plot_model(model, to_file='model.png')


                print('Test loss:', score[0])
                print('Test accuracy:', score[1])

                print ('predict', parameter_predict)

                print('C:\Deeplearning/model_ABBE_' +str(score[0])  + '.h5')

                model.summary()
