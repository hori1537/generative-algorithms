from __future__ import print_function
import random
import numpy as np
#import time
import pandas as pd
#import operator
#import math
import xlrd
import os.path
#from pprint import pprint

import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.layers import Activation

#from sklearn.model_selection import train_test_split
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from Setting_Param import ADDRESS


xlfile = "ga_setting.xlsx"
if os.path.exists(xlfile):
    print('path.exist')
    xls = xlrd.open_workbook(xlfile)
    sheet1 = xls.sheet_by_index(0)
    sheet2 = xls.sheet_by_index(1)

    #for col_index in range(sheet1.ncols):
    #    sheet1_col = sheet1.col(col_index)

    sht1_col = [sheet1.col(col_index) for col_index in range(sheet1.ncols)]
    sht2_col = [sheet2.col(col_index) for col_index in range(sheet2.ncols)]

    sht1_row = sheet1.row(0)
    sht2_row = sheet2.row(0)
    print(sht1_row)
    print(sht2_row)


    #for col_index in range(sheet2.ncols):
    #    sheet2_col = sheet2.col(col_index)

    print(sheet1_col)

    #SHEETS1
    #col[0]:index (int)					: 0,1,2,3,4,5,...
    #col[1]:name of parameter (string)	: SiO2, B2O3, Al2O3,...
    #col[2]:Priority (float)			: 1, 0.5, 1, 0, 0,...
    #col[3]:over (boolean)
    #col[4]:under (boolean)
    #col[5]:equal (boolean)
    #col[6]:target value (float)        
    #col[7]:unit of target value (string)
    
    #SHEETS2
    #col[0]:index (int)					: 0,1,2,3,4,5,...
    #col[1]:name of component (string)	: SiO2, B2O3, Al2O3,...
    #col[2]:must use(boolean)			: 1, 0, 1, 0, 0,...
    #col[3]:can use(boolean)			: 0, 1, 0, 0, 0,...
    #col[4]:better not(boolean)			: 0, 0, 0, 1, 0,...
    #col[5]:must use(boolean)			: 0, 0, 0, 0, 1,...
    #col[6]:over (boolean)
    #col[7]:under (boolean)
    #col[8]:equal (boolean)
    #col[9]:composition ratio for must use (float)



param_name = ['ABBE','DENS','FRAC','POIS','YOUN']
param_num = 5
fit_weights=[0,-1,-1,-1,0,0]
max_component_size = 62

model = [0] * param_num

for i in range(5):
    model[i]= keras.models.load_model('C:\deeplearning/model/model_' + param_name[i] + '.h5')

#param_tgt = np.empty(param_num)
#target values of parameters
#param_avg = np.empty(param_num)
#param_std = np.empty(param_num)

param_over = sheet1_col[4]
param_under = sheet1_col[5]
param_equal = sheet1_col[6]
param_tgt = sheet1_col[7]
param_avg = sheet1_col[9]
param_std = sheet1_col[10]


compo_must = sheet2_col[3]
compo_can = sheet1_col[4]
compo_better = sheet1_col[5]
compo_no = sheet1_col[6]
compo_over = sheet1_col[7]
compo_under = sheet1_col[8]
compo_equal = sheet1_col[9]
compo_tgt = sheet1_col[10]



#compo_tgt = 5     #-1

'''
param_tgt[0] = 15
param_tgt[1] = 3.227
param_tgt[2] = 0.47
param_tgt[3] = 0.28
param_tgt[4] = 90.0
'''
'''
param_avg[0] = 49.30
param_avg[1] = 2.658
param_avg[2] = 1.174
param_avg[3] = 0.22625
param_avg[4] = 90.0453

param_std[0] = 10.564
param_std[1] = 0.803472
param_std[2] = 0.63381
param_std[3] = 0.03073
param_std[4] = 48.2235
'''

#7241
# GP09-308146
#SiO2_46.42
#B2O3_12.35
#Al2O3_18.92
'''
ABBE_target= 30.6   #-1
DENS_target= 4.141  #-1
FRAC_target= 0.97   #-1
POIS_target= 0.3    #-1
YOUN_target= 90.0   #0
'''
#4770
#GI03=191757R
#SiO2   49.58
#B2O3   0
#Al2O3  10.4
#MgO
#CaO
#BaO    19.82
#LiO
#Na2O   19.2
#K2O    9.98

#[0,2,5,7,8]

#density    3.227
#Poison     0.28
#Frac       0.47



number_gene = 1000
npop=100
ngen=5
compo_list = [0,2,5,7,8,9,10,11]

creator.create("FitnessMulti", base.Fitness, weights=fit_weights)
creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

#creator.create("Individual", list, fitness=creator.FitnessMulti)


toolbox = base.Toolbox()

toolbox.register("attr_bool", np.random.choice, compo_list)
#toolbox.register("attr_bool", np.random.randint, 0, 3)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=number_gene)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def analyze_Gene(individual):
    #convert from individual to component and component_size
    
    minlength=max_component_size
    
    #individual = np.array(individual)
    
    
    component = np.bincount(individual,minlength = minlength)
    component = np.array(component)
    
    countcomponent=np.bincount(component)
    component_size = minlength - countcomponent[0]
    
    component = [i*100/number_gene for i in component]
    component = np.reshape(component, [1, 62])
    #component = np.reshape(component, [62,])
        
    return [component,component_size]


def evalProperties(component):
    #evaluate the properties from component
    
    param_predict = np.empty(param_num)
    
    for i in range(param_num):
        param_predict[i] = model[i].predict(component, batch_size = 1, verbose = 0)
        
    param_predict = np.array(param_predict)

    return param_predict

def nomalization(param):
    #parameter nomalization
    
    param_nom = np.empty(param_num)
    
    param_nom = (param - param_avg) / param_std 
    
    '''
    for i in range(param_num):
        param_nom[i] = (param[i]-param_avg[i])/param_std[i]
    
    '''
    return param_nom
    
def evalERROR(individual):
    #evaluation square error of parameters
    
    component, component_size = analyze_Gene(individual)

    param_predict = evalProperties(component)
    param_predict = np.array(param_predict)
    
    param_predict_nom = nomalization(param_predict)
    
    param_tgt_nom = nomalization(param_tgt)
    
    param_square_error = (param_predict_nom - param_tgt_nom) ** 2

    component_size_square_error = (component_size - compo_tgt) ** 2

    #for i in range(param_num):
        #print(param_name[i], '_predict is ' , param_predict[i])
        #print(param_name[i], '_square_error is ' , param_square_error[i])

    #print('component is', component)
    #print('component_size is ', component_size)
    #print('component_size_square_error ', component_size_square_error)

    return  param_square_error[0],param_square_error[1],param_square_error[2],param_square_error[3],param_square_error[4], component_size_square_error


def mutUniformINT(individual, min_ind, max_ind, indpb):
    #mutation:gene choosed from min_ind to max_ind
    
    size = len(individual)
    #print('size is ' , size)
    for i, min, max in zip(range(size), min_ind, max_ind):
        if random.random() < indpb:
            individual[i] = np.random.randint(min, max)
    return individual,


def mutUniformINTfromlist(individual,compo_list,indpb):
    #mutation:gene choosed from compo_list
    
    size = len(individual)
    #print('size is ' , size)
    for i  in range(size):
        if random.random() < indpb:
            individual[i] = np.random.choice(compo_list)
    return individual,


toolbox.register("evaluate", evalERROR)
toolbox.register("mate", tools.cxUniform,indpb=0.7)

toolbox.register("mutate", mutUniformINTfromlist,compo_list=compo_list ,indpb=0.1)
#toolbox.register("mutate", mutUniformINT, min_ind=min_ind, max_ind=max_ind, indpb=0.1)

#toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selNSGA2)
#toolbox.register("select", tools.selRoulette, k=npop)


def main():
    random.seed(64)
    pop = toolbox.population(n=npop)

    hof = tools.HallOfFame(npop, similar=np.array_equal)
    
    #hof = tools.ParetoFront(similar=np.array_equal)
    #hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    #algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, stats=stats)
    #algorithms.eaMuPlusLambda(pop, toolbox, cxpb=0.5,lambda = 5 , mutpb=0.2, ngen=1, stats=stats, halloffame=hof)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.05, ngen=ngen, stats=stats, halloffame=hof)
    
    better_ind= tools.selBest(pop,1)[0]
    better_comp, better_component_size = analyze_Gene(better_ind)
    #print(ngen ,' generation')
    #print('better comp is ', better_comp)
    #print('better component_size is ', better_component_size)
    
    param_better = evalProperties(better_comp)

    #print('better ind is ', better_ind)

    #for i in range(param_num):
    #    print(param_name[i], '_better of this pop is ' , param_better[i])

    return pop, stats, hof

if __name__ == "__main__":
    pop,stats,hof = main()
    #print('hof is', hof)
    #pprint(hof)
    hof0_comp, hof0_component_size = analyze_Gene(hof[0])

    param_hof0 = evalProperties(hof0_comp)
    print(hof0_comp)

    for i in range(param_num):
        print(param_name[i], '_better of this pop is ' ,  param_hof0[i])


    print('finish')
    