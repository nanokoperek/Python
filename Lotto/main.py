#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:13:44 2020

@author: nanokoper

"""
from lotto_src.Results import Results

#Select a day: "Tuesday", "Thursday", "Saturday"
day = "Thursday"
results = Results(day)
predicted = results.predict_all()
print(predicted)


