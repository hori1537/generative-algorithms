from __future__ import print_function
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation

from Setting_Param import ADDRESS
from sklearn.model_selection import train_test_split

import numpy as np
import time
import pandas as pd
import keras
from keras.utils import plot_model

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import operator
import math
import random

import collections

model_ABBE = keras.models.load_model('C:\deeplearning/model/model_ABBE.h5')
model_DENS = keras.models.load_model('C:\deeplearning/model/model_DENS.h5')
model_FRAC = keras.models.load_model('C:\deeplearning/model/model_FRAC.h5')
model_POIS = keras.models.load_model('C:\deeplearning/model/model_POIS.h5')
model_YOUN = keras.models.load_model('C:\deeplearning/model/model_YOUN.h5')


creator.create("FitnessMax", base.Fitness, weights=(-1.0,-1.0,-1.0,-1.0,0,-1000))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

fullcomponentsize =62

ABBE_target= 51.6
DENS_target= 4.141
FRAC_target= 0.97
POIS_target= 0.3
YOUN_target= 90.0
comp_target = 3

#7241
# GP09-308146
#SiO2_46.42
#B2O3_12.35
#Al2O3_18.92

number_gene = 100


toolbox.register("attr_bool", random.randint, 0, fullcomponentsize-1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def analyze_Gene(individual):
    minlength=fullcomponentsize
    
    component=np.bincount(individual,minlength =fullcomponentsize)
    countcomponent=np.bincount(component)
    componentsize = minlength - countcomponent[0]
    
    return [component,componentsize]

def evalProperties(individual):
    component_, componentsize_ = analyze_Gene(individual)
    component_ = np.reshape(component_, [1, 62])

    ABBE_predict = model_ABBE.predict(component_, batch_size=1, verbose=0)
    DENS_predict = model_DENS.predict(component_, batch_size=1, verbose=0)
    FRAC_predict = model_FRAC.predict(component_, batch_size=1, verbose=0)
    POIS_predict = model_POIS.predict(component_, batch_size=1, verbose=0)
    YOUN_predict = model_YOUN.predict(component_, batch_size=1, verbose=0)

    Error_ABBE = ((ABBE_predict - ABBE_target) / ABBE_target) ** 2
    Error_DENS = ((DENS_predict - DENS_target) / DENS_target) ** 2
    Error_FRAC = ((FRAC_predict - FRAC_target) / FRAC_target) ** 2
    Error_POIS = ((POIS_predict - POIS_target) / POIS_target) ** 2
    Error_YOUN = ((YOUN_predict - YOUN_target) / YOUN_target) ** 2

    Error_comp_SIZE = (componentsize_ - comp_target) ** 2

    print('ABBE predict is ', ABBE_predict)
    print('ERROR_ABBE is ',Error_ABBE)
    print('DENS predict is ', DENS_predict)
    print('ERROR_DENS is ', Error_DENS)
    print('FRAC predict is ', FRAC_predict)
    print('ERROR_FRAC is ', Error_FRAC)
    print('POIS predict is ', POIS_predict)
    print('ERROR_POIS is ', Error_POIS)
    print('YOUN predict is ', YOUN_predict)
    print('ERROR_YOUN is ', Error_YOUN)

    print('component is', component_)


    print('componentsize_ is ', componentsize_)
    print('Error_comp_SIZE ', Error_comp_SIZE)

    return Error_ABBE,Error_DENS,Error_FRAC,Error_POIS,Error_YOUN,Error_comp_SIZE


def evalABBE(individual):
    component_, componentsize_ = analyze_Gene(individual)
   # print(component_)
    component_ = np.reshape(component_,[1,62])
    ABBE_predict = model_ABBE.predict(component_, batch_size = 1, verbose=1)

   # print(component_)
    print(ABBE_predict)
    Error_ABBE = (ABBE_predict - ABBE_target)/ABBE_target**2
    return ABBE_predict


def evalDENS(individual):
    component_, componentsize_ = analyze_Gene(individual)
   # print(component_)
    component_ = np.reshape(component_,[1,62])
    DENS_predict = model_DENS.predict(component_, batch_size = 1, verbose=1)

   # print(component_)
    print(DENS_predict)
    Error_DENS = (DENS_predict - DENS_target)/DENS_target**2
    return DENS_predict

def evalFRAC(individual):
    component_, componentsize_ = analyze_Gene(individual)
   # print(component_)
    component_ = np.reshape(component_,[1,62])
    FRAC_predict = model_FRAC.predict(component_, batch_size = 1, verbose=1)

    print(FRAC_predict)
    Error_FRAC = (FRAC_predict - FRAC_target)/FRAC_target**2
    return FRAC_predict

def evalPOIS(individual):
    component_, componentsize_ = analyze_Gene(individual)
   # print(component_)
    component_ = np.reshape(component_,[1,62])
    POIS_predict = model_POIS.predict(component_, batch_size = 1, verbose=1)

    print(POIS_predict)
    Error_POIS = (POIS_predict - POIS_target)/POIS_target**2
    return POIS_predict

def evalYOUN(individual):
    component_, componentsize_ = analyze_Gene(individual)
   # print(component_)
    component_ = np.reshape(component_,[1,62])

    YOUN_predict = model_YOUN.predict(component_, batch_size = 1, verbose=1)
   # print(component_)
    print(YOUN_predict)
    Error_YOUN = (YOUN_predict - YOUN_target)/YOUN_target**2
    return YOUN_predict

def evalCOMPSIZE(individual):
    component_, componentsize_ = analyze_Gene(individual)

    Error_comp_SIZE =(componentsize_ - comp_target)**2

    return ERROR_COMPSIZE


def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2


toolbox.register("evaluate", evalProperties)
toolbox.register("mate", tools.cxUniform,indpb=0.1)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=10)

def main():
    random.seed(64)
    pop = toolbox.population(n=500)
    
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,halloffame=hof)

    best_ind= tools.selBest(pop,1)[0]
    print(best_ind)

    return pop, stats, hof

if __name__ == "__main__":
    main()

    #print('hof is', hof)