#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:53:58 2021

@author: nanokoper
"""

import pickle
import xgboost as xgb
from Data import Data
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


data_1 = Data("4", "1", "data", "dzień", 0.8, 0.11, 4)
data_2 = Data("4", "2", "data", "dzień", 0.8, 0.05, 132)
data_3 = Data("4", "3", "data", "dzień", 0.8, 0.06, 59)
data_4 = Data("4", "4", "data", "dzień", 0.8, 0.09, 39)
data_5 = Data("4", "5", "data", "dzień", 0.8, 0.06, 100)
data_6 = Data("4", "6", "data", "dzień", 0.8, 0.09, 173)

X_train_1, y_train_1, X_test_1, y_test_1 = data_1.train_test()
X_train_2, y_train_2, X_test_2, y_test_2 = data_2.train_test()
X_train_3, y_train_3, X_test_3, y_test_3 = data_3.train_test()
X_train_4, y_train_4, X_test_4, y_test_4 = data_4.train_test()
X_train_5, y_train_5, X_test_5, y_test_5 = data_5.train_test()
X_train_6, y_train_6, X_test_6, y_test_6 = data_6.train_test()

#Thursday1
DTR_1 = tree.DecisionTreeRegressor(random_state = 2)
DTR_1.fit(X_train_1, y_train_1)
filename_1 = '../lotto_model/thursday_model_1.sav'
pickle.dump(DTR_1, open(filename_1, 'wb'))

#Thursday2
DTR_2 = tree.DecisionTreeRegressor(random_state = 20)
DTR_2.fit(X_train_2, y_train_2)
filename_2 = '../lotto_model/thursday_model_2.sav'
pickle.dump(DTR_2, open(filename_2, 'wb'))

#Thursday3
ETR_3 = ExtraTreesRegressor(random_state = 0)
ETR_3.fit(X_train_3, y_train_3)
filename_3 = '../lotto_model/thursday_model_3.sav'
pickle.dump(ETR_3, open(filename_3, 'wb'))

#Thursday4
RFR_4 = RandomForestRegressor()
RFR_4.fit(X_train_4, y_train_4)
filename_4 = '../lotto_model/thursday_model_4.sav'
pickle.dump(RFR_4, open(filename_4, 'wb'))

#Thursday5
XGB_5 = xgb.XGBRFClassifier(random_state = 1)
XGB_5.fit(X_train_5, y_train_5)
filename_5 = '../lotto_model/thursday_model_5.sav'
pickle.dump(XGB_5, open(filename_5, 'wb'))

#Thursday6
DTR_6 = tree.DecisionTreeRegressor(random_state = 17)
DTR_6.fit(X_train_6, y_train_6)
filename_6 = '../lotto_model/thursday_model_6.sav'
pickle.dump(DTR_6, open(filename_6, 'wb'))