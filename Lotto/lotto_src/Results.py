#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:33:11 2021

@author: nanokoper
"""

import pickle
import numpy as np
from lotto_src.Data import Data

class Results():
    
    def __init__(self, day_to_predict):
        self.day_to_predict = day_to_predict
    
    def get_filenames(self):
        if self.day_to_predict == "Tuesday":
            filenames = []
            filename_1 = 'lotto_model/tuesday_model_1.sav'
            filename_2 = 'lotto_model/tuesday_model_2.sav'
            filename_3 = 'lotto_model/tuesday_model_3.sav'
            filename_4 = 'lotto_model/tuesday_model_4.sav'
            filename_5 = 'lotto_model/tuesday_model_5.sav'
            filename_6 = 'lotto_model/tuesday_model_6.sav'
            filenames =[filename_1, filename_2, filename_3, filename_4, filename_5, filename_6]
        
        elif self.day_to_predict == "Thursday":
            filenames = []
            filename_1 = 'lotto_model/thursday_model_1.sav'
            filename_2 = 'lotto_model/thursday_model_2.sav'
            filename_3 = 'lotto_model/thursday_model_3.sav'
            filename_4 = 'lotto_model/thursday_model_4.sav'
            filename_5 = 'lotto_model/thursday_model_5.sav'
            filename_6 = 'lotto_model/thursday_model_6.sav'
            filenames = [filename_1, filename_2, filename_3, filename_4, filename_5, filename_6]
            
        elif self.day_to_predict == "Saturday":
            filenames = []
            filename_1 = 'lotto_model/saturday_model_1.sav'
            filename_2 = 'lotto_model/saturday_model_2.sav'
            filename_3 = 'lotto_model/saturday_model_3.sav'
            filename_4 = 'lotto_model/saturday_model_4.sav'
            filename_5 = 'lotto_model/saturday_model_5.sav'
            filename_6 = 'lotto_model/saturday_model_6.sav'
            filenames = [filename_1, filename_2, filename_3, filename_4, filename_5, filename_6]
            
        else:
            filenames = None
            
        return filenames
    
    def get_for_today(self):
        
        if self.day_to_predict == "Saturday":
            for_today_1 = Data("6", "1", "data", "dzień", 1, 0.05, 139).for_today()
            for_today_2 = Data("6", "2", "data", "dzień", 1, 0.07, 119).for_today()
            for_today_3 = Data("6", "3", "data", "dzień", 1, 0.07, 119).for_today()
            for_today_4 = Data("6", "4", "data", "dzień", 1, 0.07, 119).for_today()
            for_today_5 = Data("6", "5", "data", "dzień", 1, 0.07, 119).for_today()
            for_today_6 = Data("6", "6", "data", "dzień", 1, 0.07, 119).for_today()
            for_todays = [for_today_1, for_today_2, for_today_3, for_today_4, for_today_5, for_today_6]
    
        elif self.day_to_predict == "Tuesday":
            #for_today_1 = Data("2", "1", "data", "dzień", 1, 0.19, 37).for_today()
            for_today_1 = Data("2", "1", "data", "dzień", 1, 0.19, 47).for_today()
            for_today_2 = Data("2", "2", "data", "dzień", 1, 0.2, 194).for_today()
            for_today_3 = Data("2", "3", "data", "dzień", 1, 0.06, 177).for_today()
            for_today_4 = Data("2", "4", "data", "dzień", 1, 0.08, 142).for_today()
            for_today_5 = Data("2", "5", "data", "dzień", 1, 0.14, 62).for_today()
            for_today_6 = Data("2", "6", "data", "dzień", 1, 0.2, 94).for_today()
            for_todays = [for_today_1, for_today_2, for_today_3, for_today_4, for_today_5, for_today_6]
            
        elif self.day_to_predict == "Thursday":
            for_today_1 = Data("4", "1", "data", "dzień", 0.8, 0.11, 4).for_today()
            for_today_2 = Data("4", "2", "data", "dzień", 0.8, 0.05, 132).for_today()
            for_today_3 = Data("4", "3", "data", "dzień", 0.8, 0.06, 59).for_today()
            for_today_4 = Data("4", "4", "data", "dzień", 0.8, 0.09, 39).for_today()
            for_today_5 = Data("4", "5", "data", "dzień", 0.8, 0.06, 100).for_today()
            for_today_6 = Data("4", "6", "data", "dzień", 0.8, 0.09, 173).for_today()
            for_todays = [for_today_1, for_today_2, for_today_3, for_today_4, for_today_5, for_today_6]
            
        else: 
            for_todays = None
            
        return for_todays
    
    def predict_dirty(self):
        filenames = self.get_filenames()
        for_todays = self.get_for_today()
        
        if isinstance(filenames, list) and isinstance(for_todays, list):
            predicted_results = []
            for i in range(0, 6):
                model = pickle.load(open(filenames[i], 'rb'))
                predicted = model.predict(for_todays[i])
                predicted_results.append(int(np.round(predicted[0])))
        else:
            predicted_results = "Opps! Please select another day(Thursday, Tuesday or Saturday)."
            
        return predicted_results
        
    def predict_all(self):
        predicted_dirty = self.predict_dirty()
        
        if isinstance(predicted_dirty, list):
            predicted_dirty.sort()
            for i in range(0, len(predicted_dirty) - 1):
                    if predicted_dirty[i] == predicted_dirty[i+1]:
                        if i == 0:
                            predicted_dirty[i+1] = int(np.round((predicted_dirty[i] + predicted_dirty[i+2])/2))
                        elif i == len(predicted_dirty) - 2:
                            predicted_dirty[i] = int(np.round((predicted_dirty[len(predicted_dirty) - 1] + predicted_dirty[i-1])/2))
                        else:
                            predicted_dirty[i] = int(np.round((predicted_dirty[i+1] + predicted_dirty[i-1])/2))
        
        return predicted_dirty
       