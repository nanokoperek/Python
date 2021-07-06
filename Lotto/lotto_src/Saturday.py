#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:08:18 2021

@author: nanokoper
"""

import pickle
from Data import Data
from sklearn import tree, svm



data_1 = Data("6", "1", "data", "dzień", 1, 0.05, 139)
data_2 = Data("6", "2", "data", "dzień", 1, 0.07, 119)
data_3 = Data("6", "3", "data", "dzień", 1, 0.07, 119)
data_4 = Data("6", "4", "data", "dzień", 1, 0.07, 119)
data_5 = Data("6", "5", "data", "dzień", 1, 0.07, 119)
data_6 = Data("6", "6", "data", "dzień", 1, 0.07, 119)

X_train_1, y_train_1, X_test_1, y_test_1 = data_1.train_test()
X_train_2, y_train_2, X_test_2, y_test_2 = data_2.train_test()
X_train_3, y_train_3, X_test_3, y_test_3 = data_3.train_test()
X_train_4, y_train_4, X_test_4, y_test_4 = data_4.train_test()
X_train_5, y_train_5, X_test_5, y_test_5 = data_5.train_test()
X_train_6, y_train_6, X_test_6, y_test_6 = data_6.train_test()

#Saturday1
DTR_1 = tree.DecisionTreeRegressor(random_state = 36)
DTR_1.fit(X_train_1, y_train_1)
filename_1 = '../lotto_model/saturday_model_1.sav'
pickle.dump(DTR_1, open(filename_1, 'wb'))

#Saturday2
SVR_2 = svm.SVR()
SVR_2.fit(X_train_2, y_train_2)
filename_2 = '../lotto_model/saturday_model_2.sav'
pickle.dump(SVR_2, open(filename_2, 'wb'))

from sklearn.ensemble import ExtraTreesRegressor
#Thursday3
ETR_3 = ExtraTreesRegressor(random_state = 0)
ETR_3.fit(X_train_3, y_train_3)
filename_3 = '../lotto_model/saturday_model_3.sav'
pickle.dump(ETR_3, open(filename_3, 'wb'))

from sklearn.ensemble import RandomForestRegressor
#Thursday4
RFR_4 = RandomForestRegressor()
RFR_4.fit(X_train_4, y_train_4)
filename_4 = '../lotto_model/saturday_model_4.sav'
pickle.dump(RFR_4, open(filename_4, 'wb'))

import xgboost as xgb
#Thursday5
XGB_5 = xgb.XGBRFClassifier(random_state = 1)
XGB_5.fit(X_train_5, y_train_5)
filename_5 = '../lotto_model/saturday_model_5.sav'
pickle.dump(XGB_5, open(filename_5, 'wb'))

#Thursday6
DTR_6 = tree.DecisionTreeRegressor(random_state = 17)
DTR_6.fit(X_train_6, y_train_6)
filename_6 = '../lotto_model/saturday_model_6.sav'
pickle.dump(DTR_6, open(filename_6, 'wb'))