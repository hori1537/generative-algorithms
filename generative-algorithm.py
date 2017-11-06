# -*- coding: utf-8 -*-

### set the defalut encoding 'utf-8' for pyinstaller ' making exe file'
sys.setdefaultencoding(‘utf-8’)

import random
import numpy as np
import pandas as pd

# import time
# import operator
# import math
# from pprint import pprint

import tkinter
from tkinter import filedialog
from tkinter import messagebox

import xlrd
import os
import os.path
import sys

from scipy.stats import shapiro

import keras
from keras import model
# from keras.utils import plot_model
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras import backend as K
# from keras.layers import Activation

# from sklearn.model_selection import train_test_split
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


os.chdir(os.path.dirname(sys.executable))

### default Excel file of GA setting
xlfile = "ga_setting.xlsx"
currentdirectory = os.getcwd()

'''
### choose the Excel file of GA setting with tkinter
tk = tkinter.Tk()

xlfile = filedialog.askopenfilename(initialdir=currentdirectory,
                                    title="Setting file of Generative Algorithm",
                                    filetypes=[('Excel File', '*.*')])

'''

### import parameters of generative algorithm from ga_setting.xlsx or choosen file
if os.path.exists(xlfile):
    print('path.exist')
    xls = xlrd.open_workbook(xlfile)
    sheet1 = xls.sheet_by_index(0)
    sheet2 = xls.sheet_by_index(1)
    sheet3 = xls.sheet_by_index(2)

    sht1_col = [sheet1.col(col_index) for col_index in range(sheet1.ncols)]
    sht2_col = [sheet2.col(col_index) for col_index in range(sheet2.ncols)]
    sht3_col = [sheet3.col(col_index) for col_index in range(sheet3.ncols)]

    sht1_col = [sheet1.col_values(col_index) for col_index in range(sheet1.ncols)]
    sht2_col = [sheet2.col_values(col_index) for col_index in range(sheet2.ncols)]
    sht3_col = [sheet3.col_values(col_index) for col_index in range(sheet3.ncols)]

    sht1_row = sheet1.row_values(0)
    sht2_row = sheet2.row_values(0)
    sht3_row = sheet3.row_values(0)

    # print(sht1_row)
    # print(sht2_row)
    # print(sht3_row)

    # SHEETS1
    # col[0]:index (int)					: 0,1,2,3,4,5,...
    # col[1]:name of parameter (string)	: Abbe number, Density at RT
    # col[2]:Abbreviation                : ABBE, DENS, FRAC
    # col[3]:Priority (float)			: 1, 0.5, 1, 0, 0,...
    # col[4]:over (boolean)
    # col[5]:under (boolean)
    # col[6]:equal (boolean)
    # col[7]:target value (float)
    # col[8]:unit of target value (string)

    # SHEETS2
    # col[0]:index (int)					: 0,1,2,3,4,5,...
    # col[1]:name of component (string)	: SiO2, B2O3, Al2O3,...
    # col[2]:must use(boolean)			: 1, 0, 1, 0, 0,...
    # col[3]:can use(boolean)			: 0, 1, 0, 0, 0,...
    # col[4]:better not(boolean)			: 0, 0, 0, 1, 0,...
    # col[5]:must use(boolean)			: 0, 0, 0, 0, 1,...
    # col[6]:over (boolean)
    # col[7]:under (boolean)
    # col[8]:equal (boolean)
    # col[9]:composition ratio for must use (float)

    # SHEETS3
    # col[0]:name
    # col[1]:number of gene of individual            :num_gene
    # col[2]:number of individual in population      :num_population
    # col[3]:number of populations (generations)     :num_generation
    # col[4]:proberbility of crossover of individual :cxpb
    # col[5]:proberbility of crossover of individual :cx_indpb
    # col[6]:probability of mutatio of individual    :mutpb
    # col[7]:probability of mutation of each gene    :mut_indpb

else:
    print('pass:', xlfile, ' does not exist')

### import target values of each parameter(Abbe number etc.) from xlsx
'''
'''
param_formalname    = np.array(sht1_col[1][1:6])
param_name          = np.array(sht1_col[2][1:6])
param_priority      = np.array(sht1_col[3][1:6])
param_over          = np.array(sht1_col[4][1:6])
param_under         = np.array(sht1_col[5][1:6])
param_equal         = np.array(sht1_col[6][1:6])
param_target           = np.array(sht1_col[7][1:6])
param_unit          = np.array(sht1_col[8][1:6])
param_avg           = np.array(sht1_col[9][1:6])
param_std           = np.array(sht1_col[10][1:6])

param_num           = len(param_formalname)

### import target component of glass from xlsx

component_name          = np.array(sht2_col[1][1:63])


component_must          = np.array(sht2_col[2][1:63])
component_can           = np.array(sht2_col[3][1:63])
component_better        = np.array(sht2_col[4][1:63])
component_no            = np.array(sht2_col[5][1:63])
component_over          = np.array(sht2_col[6][1:63])
component_under         = np.array(sht2_col[7][1:63])
component_equal         = np.array(sht2_col[8][1:63])
component_target           = np.array(sht2_col[9][1:63])

max_component_size  = len(component_name)


### setting values of generative algorithm from xlsx

num_gene            = sht3_col[1][3]
num_gene            = int(num_gene)
num_population      = sht3_col[2][3]
num_population      = int(num_population)
num_generation      = sht3_col[3][3]
num_generation      = int(num_generation)
cxpb                = sht3_col[4][3]
cx_indpb            = sht3_col[5][3]
mutpb               = sht3_col[6][3]
mut_indpb           = sht3_col[7][3]

is_grid_search      = sht3_col[8][3]

### fit_weights of 5 parameters from xlsx
### convert positive to negative

fit_weights = np.array(param_priority)
fit_weights = fit_weights * (-1)

### add fit_weight of component number
fit_weights = np.append(fit_weights , 0)

### add -0.0001 to the each fit_weight
### when fit_weight value is 0 cause error
fit_weights = fit_weights - 0.001
fit_weights = list(fit_weights)


component_must_and_can = np.array(component_must) + np.array(component_can)
print('component_must_and_can is ' ,component_must_and_can)

#component_list          = component_must_and_can
component_list          = np.nonzero(component_must_and_can)
component_list          = np.array(component_list)
component_list          = component_list.reshape(-1,)

component_target           = 5     #-1

'''
### default values of generative algorithm
num_gene        = 100
num_population  = 5
num_generation  = 1
cxpb            = 0.7
cx_indpb        = 0.5
mutpb           = 0.1
mut_indpb       = 0.5

is_grid_search     = 0

'''

### defalut values of setting
#param_num           = 5
#max_component_size  = 62
#component_list          = [0,2,5,7,8,9,10,11]



'''
### defalut param_target

param_name = ['ABBE','DENS','FRAC','POIS','YOUN']

param_target = np.empty(param_num)
param_avg = np.empty(param_num)
param_std = np.empty(param_num)

param_target[0] = 15
param_target[1] = 3.227
param_target[2] = 0.47
param_target[3] = 0.28
param_target[4] = 90.0

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
# 7241
# GP09-308146
# SiO2_46.42
# B2O3_12.35
# Al2O3_18.92
'''
ABBE_target= 30.6   #-1
DENS_target= 4.141  #-1
FRAC_target= 0.97   #-1
POIS_target= 0.3    #-1
YOUN_target= 90.0   #0
'''
# 4770
# GI03=191757R
# SiO2   49.58
# B2O3   0
# Al2O3  10.4
# MgO
# CaO
# BaO    19.82
# LiO
# Na2O   19.2
# K2O    9.98
# [0,2,5,7,8]
# density    3.227
# Poison     0.28
# Frac       0.47


### import predict models of deeplearning


model = [0] * param_num
for i in range(5):
    modelfile = ('C:\deeplearning/model/model_' + param_name[i] + '_n.h5')
    if os.path.exists(modelfile):
        model[i] = keras.models.load_model(modelfile)
    else:
        print(modelfile, ' does not exist. Then program exit')
        messagebox.showerror(title='error', message=(modelfile, ' does not exist. Then program exit'))
        sys.exit()


### creator
creator.create("FitnessMulti", base.Fitness, weights = fit_weights)
#creator.create("FitnessMulti", base.Fitness, weights = (-1.0,0.0001))

creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

# creator.create("Individual", list, fitness=creator.FitnessMulti)

### convert from individual to component and component_size
def analyze_Gene(individual):
    minlength = max_component_size
    # individual = np.array(individual)
    component = np.bincount(individual, minlength=minlength)
    component = np.array(component)

    countcomponent = np.bincount(component)
    component_size = minlength - countcomponent[0]

    component = [i * 100 / num_gene for i in component]
    component = np.reshape(component, [1, 62])
    # component = np.reshape(component, [62,])

    return [component, component_size]

### evaluate(predict) nomalized properties from glass component using deeplearning model by keras
def eval_nomalizedproperties(component):

    param_predict_nomalized = np.empty(param_num)

    for i in range(param_num):
        param_predict_nomalized[i] = model[i].predict(component, batch_size=1, verbose=0)

    param_predict_nomalized = np.array(param_predict_nomalized)

    return param_predict_nomalized

### nomalize the properties
def nomalization(param):
    # parameter nomalization

    param_nomalized = (param - param_avg) / param_std

    return param_nomalized


### denomalize the nomalized properties
def de_nomalization(param_nomalized):
    param = param_nomalized * param_std + param_avg
    
    return param


### evaluate the squared error:target value and predict value
def evalERROR(individual):
    # evaluation square error of parameters

    component, component_size = analyze_Gene(individual)

    param_predict_nomalized = eval_nomalizedproperties(component)
    param_predict_nomalized = np.array(param_predict_nomalized)
    param_predict           = de_nomalization(param_predict_nomalized)


    param_target_nomalized = nomalization(param_target)

    param_square_error = (param_predict_nomalized - param_target_nomalized) ** 2

    component_size_square_error = (component_size - component_target) ** 2

    #for i in range(param_num):
        #print(param_name[i], '_predict_nomalized is ' , param_predict_nomalized[i])
        #print(param_name[i], '_predict is ', param_predict[i])
        #print(param_name[i], '_square_error is ' , param_square_error[i])

    #print('component is', component)
    #print('component_size is ', component_size)
    #print('component_size_square_error ', component_size_square_error)

    return param_square_error[0], param_square_error[1], param_square_error[2], param_square_error[3], param_square_error[4], component_size_square_error


def evalshapiro(individual):
    return shapiro(individual)[1],shapiro(individual)[1]



### mutatie each gene: convert to integer
def mutUniformINT(individual, min_ind, max_ind, indpb):
    # mutation:gene choosed from min_ind to max_ind

    size = len(individual)
    # print('size is ' , size)
    for i, min, max in zip(range(size), min_ind, max_ind):
        if random.random() < indpb:
            individual[i] = np.random.randint(min, max)
    return individual,

### mutatie each gene: convert to integer in INT_list
def mutUniformINTfromlist(individual, INT_list, indpb):
    # mutation:gene choosed from component_list

    size = len(individual)
    # print('size is ' , size)
    for i in range(size):
        if random.random() < indpb:
            individual[i] = np.random.choice(INT_list)
    return individual,

### toolbox:attr_bool,individial,population

toolbox.register("attr_bool", np.random.choice, component_list)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_gene)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


### toolbox:evaluate,mate,mutate,select
toolbox.register("evaluate", evalERROR)
toolbox.register("mate", tools.cxUniform, indpb=cx_indpb)
toolbox.register("mutate", mutUniformINTfromlist, INT_list=component_list, indpb=mut_indpb)

#toolbox.register("mutate", mutUniformINTfromlist, INT_list=component_list, indpb=0.3)


#toolbox.register("select", tools.selNSGA2)
#toolbox.register("select", tools.selRoulette, k=num_population)
toolbox.register("select", tools.selTournament, tournsize=3)

def generative_algorithm():
    random.seed(64)

    pop = toolbox.population(n=num_population)
    hof = tools.HallOfFame(num_population, similar=np.array_equal)

    # hof = tools.ParetoFront(similar=np.array_equal)
    # hof = tools.HallOfFame(1)

    stats = tools.Statistics(key = lambda ind: ind.fitness.values)
    stats.register("avg",  np.mean)
    stats.register("std",  np.std)
    stats.register("min",  np.min)
    stats.register("max",  np.max)

    # algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, num_generation=1, stats=stats)
    # algorithms.eaMuPlusLambda(pop, toolbox, cxpb=0.5,lambda = 5 , mutpb=0.2, num_generation=1, stats=stats, halloffame=hof)

    print('num_gene is ', num_gene)
    print('num_pop is ', num_population)
    print('num_generation is ', num_generation)
    print('param_num is ', param_num)
    print('max_component_size is ', max_component_size)
    print('fit_weights is ', fit_weights)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.05, ngen=num_generation, stats=stats, halloffame=hof)

    better_ind = tools.selBest(pop, 1)[0]
    better_comp, better_component_size = analyze_Gene(better_ind)
    # print(num_generation ,' generation')
    # print('better comp is ', better_comp)
    # print('better component_size is ', better_component_size)

    param_better = eval_nomalizedproperties(better_comp)

    # print('better ind is ', better_ind)
    # for i in range(param_num):
    #    print(param_name[i], '_better of this pop is ' , param_better[i])

    return pop, stats, hof
    #return pop, hof


def grid_search():
    individual = np.random.choice(component_list)



    return pop, stats, hof



if __name__ == "__main__":

    if is_grid_search == 0:

        pop, stats, hof = generative_algorithm()

        #pop, hof = main()
        component_hof0, component_size_hof0 = analyze_Gene(hof[0])
        print(component_hof0)

        param_hof0_nomalized = eval_nomalizedproperties(component_hof0)
        param_hof0 =de_nomalization(param_hof0_nomalized)

        df1     = pd.DataFrame(component_name)
        df2     = pd.DataFrame(component_hof0)
        df1_2   = pd.concat([df1,df2.T],axis=1)

        df3 = pd.DataFrame(param_name)
        df4 = pd.DataFrame(param_hof0_nomalized)
        df3_4 = pd.concat([df3, df4],axis=1)

        print(df1_2)
        print(df3_4)

        df1_2.to_csv(str(currentdirectory) + '/df1_2.csv')
        df3_4.to_csv(str(currentdirectory) + '/df3_4.csv')

        df= pd.concat([df1_2,df3_4],axis=1)

        df.to_csv(str(currentdirectory) + '/hof_0.csv')


        print(str(currentdirectory) + '/hof_0.csv')
        print('fit_weights is ', fit_weights)

        for i in range(param_num):
            print(param_name[i], '_target is ', param_target[i])
            print(param_name[i], '_hall of fame is ', param_hof0[i])

        print('finish')
        #messagebox.showinfo(title = 'finish', message = 'finish')


    elif is_grid_search == 1:
        pop, stats, hof = grid_search()
