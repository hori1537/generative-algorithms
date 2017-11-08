'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import InputLayer
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

#import pydot
#import graphviz


#Date from INTERLAD

from Setting_Param_n import *
#Local_Setting_Param


raw_input = pandas.read_csv(open(str(ADDRESS)+str(CSV_NAME_DENS)))
#raw_input = numpy.loadtxt(open(str(ADDRESS)+str(CSV_NAME_DENS)), delimiter=",",skiprows=1)

[information_p, component_p, parameter_p]  = np.hsplit(raw_input, [20,82])

component = np.array(component_p)
parameter = np.array(parameter_p)


#[component_a,component]= numpy.vsplit(component_all,[0])
#[parameter_a,parameter]= numpy.vsplit(parameter_all,[0])

#print("parameter_a",parameter_a)
#print("parameter",parameter)
#print("component_a",component_a)
#print("component",component)

TRAIN_DATA_SIZE=DATA_SIZE_DENS-50
#DATE_SIZE_DENS =2448

#parameter_train, parameter_test, component_train, component_test = train_test_split(parameter, component, test_size=0.05, random_state=0)

[parameter_train, parameter_test] = numpy.vsplit(parameter, [TRAIN_DATA_SIZE])
[component_train, component_test] = numpy.vsplit(component, [TRAIN_DATA_SIZE])

batch_size = 500
epochs = 30000000
#epochs = 1


def get_model(num_layers, layer_size,bn_where,ac_last,keep_prob):
    model =Sequential()
    model.add(InputLayer(input_shape=(62,)))
    #model.add(Dense(layer_size))

    for i in range(num_layers):
        if num_layers != 1:

            model.add(Dense(layer_size))

            if bn_where==0 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

            model.add(Activation('relu'))

            if bn_where==1 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

            model.add(Dropout(keep_prob))

            if bn_where==2 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

    if ac_last ==1:
        model.add(Activation('relu'))

    model.add(Dense(1))


    model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    return model

#512,350,256,220,200,128,90

for patience_ in [1000]:

    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_, verbose=0, mode='auto')

    for num_layers in [1,3,2]:
        if num_layers !=1:
            for layer_size in[512,200]:
                for bn_where in [3,0,1,2,4]:
                    for ac_last in [0]:
                        for keep_prob in [0,0.2]:

                            model =get_model(num_layers,layer_size,bn_where,ac_last,keep_prob)

                            if layer_size >= 1024:
                                batch_size = 500
                            elif num_layers >= 4:
                                batch_size = 2000 //num_layers
                            elif bn_where ==3:
                                batch_size=500

                            else:
                                batch_size = 1000


                            model.fit(component_train, parameter_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=(component_test, parameter_test),
                                      callbacks=[es_cb])

                            score_test = model.evaluate(component_test, parameter_test, verbose=0)
                            score_train = model.evaluate(component_train, parameter_train, verbose=0)
                            test_predict = model.predict(component_test, batch_size=32, verbose=0)
                            train_predict = model.predict(component_train, batch_size=32, verbose=0)

                            df_test_pre = pd.DataFrame(test_predict)
                            df_train_pre = pd.DataFrame(train_predict)

                            df_test_param = pd.DataFrame(parameter_test)
                            df_train_param = pd.DataFrame(parameter_train)

                            df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                            df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)


                            df_dens_test.to_csv('C:\Deeplearning/model/csv/'
                                                + '_score_train-' + str("%.5f" % round(score_train[0], 5))
                                                + '_score_test-' + str("%.5f" % round(score_test[0], 5))
                                                + '_numlayer-' + str(num_layers)
                                                + '_layersize-' + str(layer_size)
                                                + '_bn_where- ' + str(bn_where)
                                                + '_ac_last-' + str(ac_last)
                                                + '_keep_prob-' + str(keep_prob)
                                                + '_patience-' + str(patience_)
                                                + '_test.csv')

                            df_dens_train.to_csv('C:\Deeplearning/model/csv/'
                                                 + '_score_train-' + str("%.5f" % round(score_train[0], 5))
                                                 + '_score_test-' + str("%.5f" % round(score_test[0], 5))
                                                 + '_numlayer-' + str(num_layers)
                                                 + '_layersize-' + str(layer_size)
                                                 + '_bn_where- ' + str(bn_where)
                                                 + '_ac_last-' + str(ac_last)
                                                 + '_keep_prob-' + str(keep_prob)
                                                 + '_patience-' + str(patience_)
                                                 + '_train.csv')



                            model.save('C:\Deeplearning/model/DENS_n/model_DENS_n'
                                            + '_score_train-'+ str("%.5f" % np.round(score_train[0], 5))
                                            +  '_score_test-'+ str("%.5f" % np.round(score_test[0], 5))
                                            + '_numlayer-' + str(num_layers)
                                            + '_layersize-' + str(layer_size)
                                            + '_bn_where- ' + str(bn_where)
                                            + '_ac_last-' + str(ac_last)
                                            + '_keep_prob-' + str(keep_prob)
                                            + '_patience-' + str(patience_)
                                            + '.h5')
                            #plot_model(model, to_file='C:\Deeplearning/model.png')
                            #plot_model(model, to_file='model.png')


                            print('Test loss:', score_test[0])
                            print('Test accuracy:', score_test[1])


                            print('C:\Deeplearning/model/DENS_n/model_DENS_n'
                                           + '_score_train-'+ str("%.5f" % round(score_train[0], 5))
                                           + '_score_test-'+ str("%.5f" % round(score_test[0], 5))
                                           + '_numlayer-' + str(num_layers)
                                           + '_layersize-' + str(layer_size)
                                           + '_bn_where- ' + str(bn_where)
                                           + '_ac_last-' + str(ac_last)
                                           + '_keep_prob-' + str(keep_prob)
                                           + '_patience-' + str(patience_)
                                           + '.h5')


                            model.summary()

        else:
            layer_size=62
            bn_where=1
            keep_prob=0.2
            for ac_last in [1]:
                model = get_model(num_layers, layer_size, bn_where, ac_last, keep_prob)

                model.fit(component_train, parameter_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(component_test, parameter_test),
                          callbacks=[es_cb])

                score_test = model.evaluate(component_test, parameter_test, verbose=0)
                score_train = model.evaluate(component_train, parameter_train, verbose=0)
                test_predict = model.predict(component_test, batch_size=32, verbose=1)
                train_predict = model.predict(component_train, batch_size=32, verbose=1)

                df_test_pre = pd.DataFrame(test_predict)
                df_train_pre = pd.DataFrame(train_predict)

                df_test_param = pd.DataFrame(parameter_test)
                df_train_param = pd.DataFrame(parameter_train)

                df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)



                df_dens_test.to_csv('C:\Deeplearning/model/csv/'
                                       +'_score_train-'+ str("%.5f" % round(score_train[0], 5))
                                       + '_score_test-'+ str("%.5f" % round(score_test[0], 5))
                                       + '_numlayer-' + str(num_layers)
                                       + '_layersize-' + str(layer_size)
                                       + '_bn_where- ' + str(bn_where)
                                       + '_ac_last-' + str(ac_last)
                                       + '_keep_prob-' + str(keep_prob)
                                       + '_patience-' + str(patience_)
                                       + '_test.csv')

                df_dens_train.to_csv('C:\Deeplearning/model/csv/'
                                       +'_score_train-'+ str("%.5f" % round(score_train[0], 5))
                                       + '_score_test-'+ str("%.5f" % round(score_test[0], 5))
                                       + '_numlayer-' + str(num_layers)
                                       + '_layersize-' + str(layer_size)
                                       + '_bn_where- ' + str(bn_where)
                                       + '_ac_last-' + str(ac_last)
                                       + '_keep_prob-' + str(keep_prob)
                                       + '_patience-' + str(patience_)
                                       + '_train.csv')


                model.save('C:\Deeplearning/model/DENS_n/model_DENS_n'
                                       + '_score_train-'+ str("%.5f" % round(score_train[0], 5))
                                       + '_score_test-'+ str("%.5f" % round(score_test[0], 5))
                                       + '_numlayer-' + str(num_layers)
                                       + '_layersize-' + str(layer_size)
                                       + '_bn_where- ' + str(bn_where)
                                       + '_ac_last-' + str(ac_last)
                                       + '_keep_prob-' + str(keep_prob)
                                       + '_patience-' + str(patience_)
                                       + '.h5')
                # plot_model(model, to_file='C:\Deeplearning/model.png')
                #plot_model(model, to_file='model.png')


                print('Test loss:', score_test[0])
                print('Test accuracy:', score_test[1])

                #print('predict', test_predict)

                print('C:\Deeplearning/model/DENS_n/model_DENS_n'
                               + '_score_train-'+ str("%.5f" % round(score_train[0], 5))
                               + '_score_test-'+ str("%.5f" % round(score_test[0], 5))
                               + '_numlayer-' + str(num_layers)
                               + '_layersize-' + str(layer_size)
                               + '_bn_where- ' + str(bn_where)
                               + '_ac_last-' + str(ac_last)
                               + '_keep_prob-' + str(keep_prob)
                               + '_patience-' + str(patience_)
                               + '.h5')





                model.summary()