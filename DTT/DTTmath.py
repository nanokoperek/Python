import pandas as pd
import statistics as st
import numpy as np
from dateutil.parser import parse


class Data:
    
    def __init__(self, data_path, data_type, data_subject):
        self.data_path = data_path
        self.data_type = data_type
        self.data_subject = data_subject
        self.data_frame = pd.read_csv(data_path)
        self.columns = list(self.data_frame.columns)
        
    def __remove_empty_list__(list_dirty):
        list_clean = []
        for dirty in list_dirty:
            if len(dirty) != 0:
                list_clean.append(dirty)
        return list_clean
    
#Roznica miedzy min i max:
    def __diff_minmax__(min_value, max_value):
        return abs(max_value - min_value)
    
    def is_date(self, fuzzy=False):
        is_date_columns = []
        is_date = False
        for column in self.columns:
            try: 
                parse(str(self.data_frame[column][0]), fuzzy=fuzzy)
                is_date_columns.append(True)
            except ValueError:
                is_date_columns.append(False)
        if True in is_date_columns:
            is_date = True
        return is_date

    def find_global_min(self):
        min_columns = {}
        for column in self.columns:
            min_date = []
            if type(self.data_frame[column][0]) == np.float64:
                min_date.append(self.data_frame[column].min())
                min_columns[column] = min_date
        return min_columns

    def find_global_max(self):
        max_columns = {}
        for column in self.columns:
            max_date = []
            if type(self.data_frame[column][0]) == np.float64:
                max_date.append(self.data_frame[column].max())
                max_columns[column] = max_date
        return max_columns

#Zczytanie labela z kolumny X i Y
    def column_names(self):
        column_names = []
        for column in self.columns:
            if column.find("[")!=-1 or column.find("]")!=-1:
                j = column[:column.find("[")].strip()
                column_names.append(j)
            else: 
                column_names.append(column)
        return column_names

#Zczytanie jednostek z labeli
    def column_units(self):
        column_units = []
        for column in self.columns:
            if column.find("[")==-1 or column.find("]")==-1:
                column_units.append('None')
            else:
                j = column[column.find("[")+1:-1].strip()
                column_units.append(j)
        return column_units

#Znalezienie top 5 i bottom 5:
    def find_top(self, number_top_rows = 5):
        sorted_df = self.data_frame.sort_values(self.data_frame.columns[0], ascending = False)
        return sorted_df.head(number_top_rows)
    
    def find_bottom(self, number_bottom_rows = 5):
        sorted_df = self.data_frame.sort_values(self.data_frame.columns[0])
        return sorted_df.head(number_bottom_rows)

#Tendencja globalnie(pierwszy i ostatni punkt) i lokalnie(przegięcia):
    def tendency_global(self):
        sorted_df = self.data_frame.sort_values(self.columns[1])
        tendency = list()
        if sorted_df[sorted_df.columns[0]][0] < sorted_df[sorted_df.columns[0]][sorted_df.shape[0]-1]:
            tendency.extend(['increase', sorted_df[sorted_df.columns[1]][0], sorted_df[sorted_df.columns[1]][sorted_df.shape[0]-1]])
        elif sorted_df[sorted_df.columns[0]][0] > sorted_df[sorted_df.columns[0]][sorted_df.shape[0]-1]:
            tendency.extend(['decrease', sorted_df[sorted_df.columns[1]][0], sorted_df[sorted_df.columns[1]][sorted_df.shape[0]-1]])
        else:
            tendency.extend(['flat', sorted_df[sorted_df.columns[1]][0], sorted_df[sorted_df.columns[1]][sorted_df.shape[0]-1]])
        return tendency

    def add_tendency_to_df(self):
        sorted_df = self.data_frame.sort_values(self.columns[1])
        tendency = list()
        for i in range(0, sorted_df.shape[0]):
            if i == 0:
                tendency.append('None')
            else:
                if (sorted_df[sorted_df.columns[0]][i] < sorted_df[sorted_df.columns[0]][i-1]) and i != 0 :
                    tendency.append('decrease')
                elif sorted_df[sorted_df.columns[0]][i] > sorted_df[sorted_df.columns[0]][i-1] and i != 0:
                    tendency.append('increase')
                else:
                    tendency.append('flat')
        self.data_frame['tendency'] = tendency


    def tendency_period(self):
        sorted_df = self.data_frame.sort_values(self.columns[1])
        periods_start_date = list()
        periods_end_date = list()
        
        for i in range(1, sorted_df.shape[0]-1):
            tendency_periods_single = list()
            if sorted_df[sorted_df.columns[3]][i] != sorted_df[sorted_df.columns[3]][i-1]:
                tendency_periods_single.append(sorted_df[sorted_df.columns[3]][i])
                tendency_periods_single.append(sorted_df[sorted_df.columns[1]][i])
            periods_start_date.append(tendency_periods_single)
            
        for i in range(1, sorted_df.shape[0]):
            if i != sorted_df.shape[0]-1:
                if sorted_df[sorted_df.columns[3]][i] != sorted_df[sorted_df.columns[3]][i+1]:
                    periods_end_date.append(sorted_df[sorted_df.columns[1]][i])
    
        periods_start_date = self.__remove_empty_list__(periods_start_date)
        
        for i in range(0, len(periods_start_date)):
            periods_start_date[i].append(periods_end_date[i])
    
        return periods_start_date
    
#Min/max lokalne:
    def get_minmax_local(self):
        sorted_df = self.data_frame.sort_values(self.columns[1])
        list_minmax_local = list()
        for i in range(1, sorted_df.shape[0]-1):
            minmax_local_single=list()
            if sorted_df[sorted_df.columns[0]][i] < sorted_df[sorted_df.columns[0]][i+1] and sorted_df[sorted_df.columns[0]][i] < sorted_df[sorted_df.columns[0]][i-1]:
                minmax_local_single.extend(['minimum', sorted_df[sorted_df.columns[0]][i], sorted_df[sorted_df.columns[1]][i]])
                list_minmax_local.append(minmax_local_single)
            elif sorted_df[sorted_df.columns[0]][i] > sorted_df[sorted_df.columns[0]][i+1] and sorted_df[sorted_df.columns[0]][i] > sorted_df[sorted_df.columns[0]][i-1]:
                minmax_local_single.extend(['maximum', sorted_df[sorted_df.columns[0]][i], sorted_df[sorted_df.columns[1]][i]])
                list_minmax_local.append(minmax_local_single)
        return list_minmax_local
#XoX:
    def xox(self):
        sorted_df = self.data_frame.sort_values(self.columns[1])
        self.data_frame['xox'] = sorted_df[sorted_df.columns[0]].pct_change()

    
#Ilosc kolumn:
    def number_columns(self):
        return len(self.columns)

#Suma, srednia, mediana
    def sum_all_period(self):
        column = self.data_frame[self.columns[0]]
        return sum(column)

    def avg_all_period(self):
        column = self.data_frame[self.columns[0]]
        return st.mean(column)

    def median_all_period(self):
        sorted_df = self.data_frame.sort_values(self.columns[0])
        column = sorted_df[sorted_df.columns[0]]
        return st.median(column)

#Percentyle
    def percentile(self, prc):
        sorted_df = self.data_frame.sort_values(self.columns[0])
        column = sorted_df[sorted_df.columns[0]]
        print(np.percentile(column, prc))
    

