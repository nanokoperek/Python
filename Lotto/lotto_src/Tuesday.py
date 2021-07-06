#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:33:44 2021

@author: nanokoper
"""

import pickle
from lotto_src.Data import Data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier 
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier


data_1 = Data("2", "1", "data", "dzień", 1, 0.19, 47)
data_2 = Data("2", "2", "data", "dzień", 1, 0.2, 194)
data_3 = Data("2", "3", "data", "dzień", 1, 0.06, 177)
data_4 = Data("2", "4", "data", "dzień", 1, 0.08, 142)
data_5 = Data("2", "5", "data", "dzień", 1, 0.14, 62)
data_6 = Data("2", "6", "data", "dzień", 1, 0.2, 94)

X_train_1, y_train_1, X_test_1, y_test_1 = data_1.train_test()
X_train_2, y_train_2, X_test_2, y_test_2 = data_2.train_test()
X_train_3, y_train_3, X_test_3, y_test_3 = data_3.train_test()
X_train_4, y_train_4, X_test_4, y_test_4 = data_4.train_test()
X_train_5, y_train_5, X_test_5, y_test_5 = data_5.train_test()
X_train_6, y_train_6, X_test_6, y_test_6 = data_6.train_test()

#Tuesday1
"""
OvR_1 = OneVsRestClassifier(RandomForestClassifier(random_state=0))
OvR_1.fit(X_train_1, y_train_1)
filename_1 = '../lotto_model/tuesday_model_1.sav'
pickle.dump(OvR_1, open(filename_1, 'wb'))
"""
OvR_GBR_1 = OneVsRestClassifier(GradientBoostingRegressor(random_state=0))
OvR_GBR_1.fit(X_train_1, y_train_1)
filename_1 = '../lotto_model/tuesday_model_1.sav'
pickle.dump(OvR_GBR_1, open(filename_1, 'wb'))

#Tuesday2

GBR_2 = GradientBoostingRegressor(random_state=0)
GBR_2.fit(X_train_2, y_train_2)
filename_2 = '../lotto_model/tuesday_model_2.sav'
pickle.dump(GBR_2, open(filename_2, 'wb'))
"""
from sklearn.ensemble import RandomForestRegressor
RFR_2 = RandomForestRegressor()
RFR_2.fit(X_train_2, y_train_2)
filename_2 = '../lotto_model/tuesday_model_2.sav'
pickle.dump(RFR_2, open(filename_2, 'wb'))
"""
#Tuesday3

GBR_3 = GradientBoostingRegressor(random_state=0)
GBR_3.fit(X_train_3, y_train_3)
filename_3 = '../lotto_model/tuesday_model_3.sav'
pickle.dump(GBR_3, open(filename_3, 'wb'))

#Tuesday4
"""
RFR_4 = RandomForestRegressor()
RFR_4.fit(X_train_4, y_train_4)
filename_4 = '../lotto_model/tuesday_model_4.sav'
pickle.dump(RFR_4, open(filename_4, 'wb'))

from sklearn.neighbors import KNeighborsRegressor
NR_4 = KNeighborsRegressor()
NR_4.fit(X_train_4, y_train_4)
filename_4 = '../lotto_model/tuesday_model_4.sav'
pickle.dump(NR_4, open(filename_4, 'wb'))

ETR_4 = ExtraTreesRegressor(random_state=0)
ETR_4.fit(X_train_4, y_train_4)
filename_4 = '../lotto_model/tuesday_model_4.sav'
pickle.dump(ETR_4, open(filename_4, 'wb'))
"""
GBR_4 = GradientBoostingRegressor(random_state=0)
GBR_4.fit(X_train_4, y_train_4)
filename_4 = '../lotto_model/tuesday_model_4.sav'
pickle.dump(GBR_4, open(filename_4, 'wb'))

#Tuesday5
"""
RFR_5 = RandomForestRegressor()
RFR_5.fit(X_train_5, y_train_5)
filename_5 = '../lotto_model/tuesday_model_5.sav'
pickle.dump(RFR_5, open(filename_5, 'wb'))
"""
from sklearn.ensemble import GradientBoostingClassifier
GBC_5 = GradientBoostingClassifier(random_state=0).fit(X_train_5, y_train_5)
filename_5 = '../lotto_model/tuesday_model_5.sav'
pickle.dump(GBC_5, open(filename_5, 'wb'))

"""
DTC_5 = tree.DecisionTreeClassifier(random_state=0)
DTC_5.fit(X_train_5, y_train_5)
filename_5 = '../lotto_model/tuesday_model_5.sav'
pickle.dump(DTC_5, open(filename_5, 'wb'))
"""
#Tuesday6
import xgboost as xgb
XGB_6 = xgb.XGBRFClassifier()
XGB_6.fit(X_train_6, y_train_6)
filename_6 = '../lotto_model/tuesday_model_6.sav'
pickle.dump(XGB_6, open(filename_6, 'wb'))