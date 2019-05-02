#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:01:21 2019
@author: nanokoper
"""
import math
import numpy as npimport 
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import numpy as np

def fibo(k = 70, m = 100):
   count=0
   a = 0  # x(n-1)
   b = 1  # x(n)
   t = 1
   ciag = []    
   while count < k:        
       t = a
       a = (a+b) % m
       b = t
       count = count+1
       ciag.append(a)
   return ciag

def fibo_del():
   m = 100
   p = 24  #k
   q = 55 #l
   # współczynnik określający zakres generowanych liczb pseudolosowych (od 0 do m-1)
   ciag_wyjsciowy = []
   dl_ciagu = 70
   #initial_values = [8, 6000, 7000, 5000, 3000, 12121, 90000]  # q = len(initial_values)
   initial_values = []
   for v in range(q):
       v = datetime.now().microsecond % m
       initial_values.append(v)
   for n in range(dl_ciagu):
       for i in range(len(initial_values)):

           if i is 0:
               out = (initial_values[p - 1] + initial_values[q - 1]) % m  # the pseudorandom output
           elif 0 < i < len(initial_values) - 1:
               initial_values[i] = initial_values[i + 1]  # shift the array
           else:
               initial_values[i] = out
               ciag_wyjsciowy.append(initial_values[i])
               # i-=1
   return ciag_wyjsciowy

def numppp():  # generator numpy
   m = 100
   dl_ciagu = 70
   ciag_wyjsciowy = []
   for n in range(dl_ciagu):
       ciag_wyjsciowy.append(int(npimport.random.random() * m))
   return ciag_wyjsciowy

# def monte_carlo(input_numbers, m):
#    success = 0 #proby udane
#    trials = 0  #wszysttkie proby
#    for i in range(0, len(input_numbers), 2):
#        x = (input_numbers[i] / m) * 2 - 1
#        y = (input_numbers[i + 1] / m) * 2 - 1
#        result = is_inside_circle(x, y)
#        if result == True:
#            success = success + 1
#        trials = trials + 1
#    our_ratio = success / trials  # stosunek prob trafionych do wszystkich
#    right_ratio = math.pi / 4     # stosunek pola kola do pola kwadratu, w ktory jest  wpisane    print("ratio for given numbers = ", our_ratio)
#    print("right ratio = ", right_ratio)
#    print("dif = ", abs(right_ratio - our_ratio))
#
# def is_inside_circle(x, y):
#    radius = 1
#    center_x = 0
#    center_y = 0
#    return ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) < (radius * radius)#input_numbers = fibonacci_del(10000)

#KS: Two samples have the same distribution, pvalue - probablity of observing the stat D

def stat_tests_ks1(zbior):
    D,p = stats.kstest(zbior, 'uniform',args=(min(zbior),max(zbior)))
    if p < 0.05: # odrzucamy hipoteze 0
       wynik = 0
    else:
       wynik = 1
    return wynik

def stat_tests_ks2(zbior1,zbior2):
    D, p = stats.ks_2samp(zbior1, zbior2)
    if p < 0.05:
        wynik = 0
    else:
        wynik = 1
    return wynik

def wykresiki(fibosimpl, fibodel, nampi):
    
   fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 10))
   fig.suptitle("Histogramy", fontsize=18, ha = 'center')
   axs[0].hist(fibosimpl, 20, color = "red", ec="black")
   axs[0].set_xlabel("FiboSimple")
   axs[1].hist(fibodel, 20, color = "green", ec='black')
   axs[1].set_xlabel("FiboDel")
   axs[2].hist(nampi, 20, color = "blue", ec='black')
   axs[2].set_xlabel("RNumpy")
   plt.show()

input_num = numppp()

n = 50
result = 0

### testy dla n prób
### fibonaccie_del
for i in range(n):
    i = fibo_del()
    stat_tests_ks1(i)
    result+=stat_tests_ks1(i)
print('Fibonacci Opóźniony: ',result*100/n)

result = 0
### fibonaccie
for i in range(n):
    i = fibo()
    stat_tests_ks1(i)
    result+=stat_tests_ks1(i)
print('Fibonacci: ',result*100/n)

result = 0
### numppy
for i in range(n):
    i = numppp()
    stat_tests_ks1(i)
    result+=stat_tests_ks1(i)
print('Numpy: ',result*100/n)


# a = stat_tests_ks1(fibo())
# b = stat_tests_ks1(fibo_del())
# c = stat_tests_ks1(numppp())
# d = stat_tests_ks2(fibo(),fibo_del())
# e = stat_tests_ks2(fibo(),numppp())
# f = stat_tests_ks2(fibo_del(),numppp())
#
# print()
# print("fibosimpl, 'uniform': ", a)
# print("fibodel, 'uniform': ", b)
# print("nampi, 'uniform': ", c)
# print("fibosimpl, fibodel: ", d)
# print("fibosimpl, nampi: ", e)
# print("fibodel, nampi: ", f)

wykresiki(fibo(),fibo_del(), numppp())
