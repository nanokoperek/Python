# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import statistics as st
import numpy as np
from dateutil.parser import parse

#Czytanie danych - czyta te kolumny, gdzie cos jest
df = pd.read_csv("dane.csv")

#Sprawdzenie czy dataset ma daty
def is_date(string, fuzzy=False):
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

#Znalezienie min/max
def findGlobalMin(column):
    globalMin = column.min()
    return globalMin

def findGlobalMax(column):
    globalMax = column.max()
    return globalMax

#Podanie tematyki danych
def data_intro():
    data_intro = dict()
    #data_type = input("Data type to present (data/graph/chart/diagram etc.):")
    #data_subject = input("Data subject:")
    data_type = 'data'
    data_subject = 'money spent on iphones in the UK'
    data_is_date = is_date(str(df[df.columns[1]][0]))
    data_intro['data_type'] = data_type
    data_intro['data_subject'] = data_subject
    data_intro['is_date'] = data_is_date
    if data_intro['is_date'] == True:
        data_intro['start_date'] = findGlobalMin(df[df.columns[1]])
        data_intro['end_date'] = findGlobalMax(df[df.columns[1]])
    return data_intro

#Zczytanie labela z kolumny X i Y
def columnNames(dataset):
    columnNames = list(dataset.columns)
    for i in columnNames:
        if i.find("[")!=-1 or i.find("]")!=-1:
            j = i[:i.find("[")].strip()
            columnNames[columnNames.index(i)]=j
    return columnNames

#Zczytanie jednostek z labeli
def columnUnits(dataset):
    columnNames = list(dataset.columns)
    columnUnits = []
    for i in columnNames:
        if i.find("[")==-1 or i.find("]")==-1:
            columnUnits.append('None')
        else:
            j = i[i.find("[")+1:-1].strip()
            columnUnits.append(j)
    return columnUnits

#Znalezienie top 5 i bottom 5:
def findTopX(dataset, numberTopRows = 5):
    sortedDf = dataset.sort_values(dataset.columns[0], ascending = False)
    return sortedDf.head(numberTopRows)

def findBottomX(dataset, numberBottomRows = 5):
    sortedDf = dataset.sort_values(dataset.columns[0])
    return sortedDf.head(numberBottomRows)

#Tendencja globalnie(pierwszy i ostatni punkt) i lokalnie(przegięcia):
def tendencyGlobal(dataset):
    sortedDf = dataset.sort_values(dataset.columns[1])
    tendency = list()
    if sortedDf[sortedDf.columns[0]][0] < sortedDf[sortedDf.columns[0]][sortedDf.shape[0]-1]:
        tendency.extend(['increase', sortedDf[sortedDf.columns[1]][0], sortedDf[sortedDf.columns[1]][sortedDf.shape[0]-1]])
    elif sortedDf[sortedDf.columns[0]][0] > sortedDf[sortedDf.columns[0]][sortedDf.shape[0]-1]:
        tendency.extend(['decrease', sortedDf[sortedDf.columns[1]][0], sortedDf[sortedDf.columns[1]][sortedDf.shape[0]-1]])
    else:
        tendency.extend(['flat', sortedDf[sortedDf.columns[1]][0], sortedDf[sortedDf.columns[1]][sortedDf.shape[0]-1]])
    return tendency

            
def addTendencyToDf(dataset):
    sortedDf = dataset.sort_values(dataset.columns[1])
    tendency = list()
    for i in range(0, sortedDf.shape[0]):
        if i == 0:
            tendency.append('None')
        else:
            if (sortedDf[sortedDf.columns[0]][i] < sortedDf[sortedDf.columns[0]][i-1]) and i != 0 :
                tendency.append('decrease')
            elif sortedDf[sortedDf.columns[0]][i] > sortedDf[sortedDf.columns[0]][i-1] and i != 0:
                tendency.append('increase')
            else:
                tendency.append('flat')
    sortedDf['tendency'] = tendency
    return sortedDf

def tendencyPeriod(dataset):
    sortedDf = dataset.sort_values(dataset.columns[1])
    periodsStartDate = list()
    periodsEndDate = list()
    
    for i in range(1, sortedDf.shape[0]-1):
        tendencyPeriodsSingle = list()
        if sortedDf[sortedDf.columns[2]][i] != sortedDf[sortedDf.columns[2]][i-1]:
            tendencyPeriodsSingle.append(sortedDf[sortedDf.columns[2]][i])
            tendencyPeriodsSingle.append(sortedDf[sortedDf.columns[1]][i])
        periodsStartDate.append(tendencyPeriodsSingle)
        
    for i in range(1, sortedDf.shape[0]):
        if i != sortedDf.shape[0]-1:
            if sortedDf[sortedDf.columns[2]][i] != sortedDf[sortedDf.columns[2]][i+1]:
                periodsEndDate.append(sortedDf[sortedDf.columns[1]][i])

    periodsStartDate = removeEmptyList(periodsStartDate)
    
    for i in range(0, len(periodsStartDate)):
        periodsStartDate[i].append(periodsEndDate[i])

    return periodsStartDate

def removeEmptyList(listDirty):
    listClean = list()
    for i in listDirty:
        if len(i) != 0:
            listClean.append(i)
    return listClean

#Min/max lokalne:
def getMinMaxLocal(dataset):
    sortedDf = dataset.sort_values(dataset.columns[1])
    listMinMaxLocal = list()
    for i in range(1, sortedDf.shape[0]-1):
        minMaxLocalSingle=list()
        if sortedDf[sortedDf.columns[0]][i] < sortedDf[sortedDf.columns[0]][i+1] and sortedDf[sortedDf.columns[0]][i] < sortedDf[sortedDf.columns[0]][i-1]:
            minMaxLocalSingle.extend(['minimum', sortedDf[sortedDf.columns[0]][i], sortedDf[sortedDf.columns[1]][i]])
            listMinMaxLocal.append(minMaxLocalSingle)
        elif sortedDf[sortedDf.columns[0]][i] > sortedDf[sortedDf.columns[0]][i+1] and sortedDf[sortedDf.columns[0]][i] > sortedDf[sortedDf.columns[0]][i-1]:
            minMaxLocalSingle.extend(['maximum', sortedDf[sortedDf.columns[0]][i], sortedDf[sortedDf.columns[1]][i]])
            listMinMaxLocal.append(minMaxLocalSingle)
    return listMinMaxLocal
    
#XoX:
def xox(dataset):
    sortedDf = dataset.sort_values(dataset.columns[1])
    sortedDf['xox'] = sortedDf[sortedDf.columns[0]].pct_change()
    return sortedDf
    
#Ilosc kolumn:
def numberColumns(dataset):
    return len(dataset.columns)

#Suma, srednia, mediana
def sumAllPeriod(dataset):
    column = dataset[dataset.columns[0]]
    return sum(column)


def avgAllPeriod(dataset):
    column = dataset[dataset.columns[0]]
    return st.mean(column)

def medianAllPeriod(dataset):
    sortedDf = dataset.sort_values(dataset.columns[0])
    column = sortedDf[sortedDf.columns[0]]
    return st.median(column)

#Percentyle
def percentile(dataset, prc):
    sortedDf = dataset.sort_values(dataset.columns[0])
    column = sortedDf[sortedDf.columns[0]]
    print(np.percentile(column, prc))
    
#Roznica miedzy min i max:
def diffMinMax(minValue, maxValue):
    return abs(maxValue - minValue)
