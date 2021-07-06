#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:56:17 2020

@author: nanokoper
"""
import pandas as pd
import pandasql as ps
import numpy as np

class Data():
    
    def __init__(self, day, name_column, date_column, day_column, train_size, perc, seq_len, file_name = None, data_path = None):
        self.day = day
        self.name_column = name_column
        self.date_column = date_column
        self.day_column = day_column
        self.train_size = train_size
        self.perc = perc
        self.seq_len = seq_len
        self.file_name = 'full_data.csv'
        self.data_path = 'lotto_data/'
        self.df = pd.read_csv(self.data_path + self.file_name)


    def filter_by_day(self, df_raw):
        day_query = "SELECT * FROM df_raw where dzień == {};"
        day_query = day_query.format("'" + self.day + "'")
        df = ps.sqldf(day_query)
        return df
    
    def filter_by_column(self, df_raw):
        df = df_raw[self.name_column]
        return df

    def train_test(self):
        test_size = 1 - self.train_size
        df_day = self.filter_by_day(self.df)
        df_column_day = self.filter_by_column(df_day)
        seq=self.get_sequences(df_column_day, self.seq_len)
        X, y = seq[:, :-1], seq[:,-1]
        X_train = X[:int(X.shape[0] * self.train_size)]
        y_train = y[:int(y.shape[0] * self.train_size)]
        X = X[::-1]
        y = y[::-1]
        X_test = X[:int(X.shape[0] * test_size)]
        y_test = y[:int(y.shape[0] * test_size)]
        X_test = X_test[::-1]
        y_test = y_test[::-1]
        return X_train, y_train, X_test, y_test
    
    def for_today(self):
        df_day = self.filter_by_day(self.df)
        df_column_day = self.filter_by_column(df_day)
        for_today = df_column_day[-self.seq_len:]
        for_today = for_today.to_numpy()
        for_today = for_today.reshape(len(for_today), 1).T
        return for_today
    

    @staticmethod
    def get_sequences(values, no_preval):
        no_preval = no_preval + 1
        seq = []
        seq_temp = []
        for i in range(len(values) - no_preval + 1):
            seq_temp = list(values[i:no_preval+i])
            seq.append(seq_temp)
        array = np.array(seq)
        return array
    