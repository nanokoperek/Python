# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:59:23 2020

"""

import string
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from DTTmath import data_intro


x = data_intro()

#Przygotowanie zdania inicjującego (seed):

#init_sent_open = open("initialPolski.txt", "r")
#init_sent = init_sent_open.read()

init_sent_open = open("initialEnglish.txt", "r")
init_sent = init_sent_open.read()

def pickInitSent(text_file):
    text = text_file.split("\n")
    random.shuffle(text)
    init_sent_picked = random.choice(text)
    return init_sent_picked

def countWords(text):
    list_words = text.split(" ")
    n_words = len(list_words)
    return n_words

seed_text_dirty = pickInitSent(init_sent)
j = countWords(seed_text_dirty)

#Przygotowanie tekstu do trenowania modelu:

text_to_train_open = open("intro.txt", "r")
text_to_train = text_to_train_open.read()

def cleanText(docDirty):
    tokens = docDirty.split()
    table = str.maketrans('','', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

tokens = cleanText(text_to_train)
seed_text = cleanText(seed_text_dirty)
seed_text = ' '.join(seed_text)

#Na inputcie jest 50 słów a przewidywane jest 1 słowo; ta część tworzy linie tekstu
#gdzie każda jest dłuższa o jedno słowo i ostatnie słowo z poprzedniej linii jest przewidywane
#length 50 + 1

def generate_lines(length):
    lines = []

    for i in range(length, len(tokens)):
        seq = tokens[i-length:i]
        line = ' '.join(seq)
        lines.append(line)
        if i > 200000:
            break
    return lines
    

#Tokenizacja - zamiana słów na intigery
lines = generate_lines(j+1)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
sequences = np.array(sequences)

X, y = sequences[:, :-1], sequences[:,-1]
vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes = vocab_size)
seq_length = X.shape[1]


#LSTM Model
hidden_size = 100

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size = 256, epochs = 40)

#Generacja tekstu
def generate_next_seq(model, tokenizer, text_seq_length, seed_text, n_words):
    text=[]
    
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')
        y_predict = model.predict_classes(encoded)

        next_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_predict:
                next_word = word
                break
        if (next_word == 'datasubject' and x['is_date'] == False) or next_word == 'enddate':
            seed_text = seed_text + ' ' + next_word
            text.append(next_word + '.')
            break
        else:
            seed_text = seed_text + ' ' + next_word
            text.append(next_word)
    return ' '.join(text)


def replace_type_subject(data_type, data_subject, text_generated, data_intro):
    text_generated = text_generated.replace("datatype", data_type)
    text_generated = text_generated.replace("datasubject", data_subject)
    if 'start_date' in data_intro:
        text_generated = text_generated.replace("startdate", str(data_intro['start_date']))
    if 'end_date' in data_intro:
        text_generated = text_generated.replace("enddate", str(data_intro['end_date']))      
    return text_generated

text_generated = generate_next_seq(model, tokenizer, seq_length, seed_text, 20)

"""   
if x['is_date'] == True:
    seed_text.append('startdate')
    seed_text.append('enddate')

def get_seed_after_stop(text, stop_word):
    stop_word_list = []
    text_list = text.split()
    
    for word in text_list:
        stop_word_line = []
        if word == stop_word:
            word_1 = text_list[text_list.index(stop_word) + 1]
            word_2 = text_list[text_list.index(stop_word) + 2]
            stop_word_line.append(word_1)
            stop_word_line.append(word_2)
            stop_word_line = ' '.join(stop_word_line)
        if len(stop_word_line) > 0:
            stop_word_list.append(stop_word_line)
    picked_one = random.choice(stop_word_list)
    return picked_one


text_generated = text_generated + " " + text_after_stop
text_generated = replace_type_subject(x['data_type'], x['data_subject'], text_generated, x)

print(text_generated)


def back_forward_propagation(seed_text):
    text_to_enddate_subject = generate_next_seq(model, tokenizer, text_seq_length, seed_text, n_words)
    return seed_text[0] + text_to_enddate_subject


file1 = open("test.txt","a")#append mode 
if text_generated != "the largest value was maxvalue and it was in totalmaxdate the smallest value was minvalue and it was in totalmindate":
    file1.write(seed_text + ". " + text_generated + "\n" + "\n") 
file1.close() 
"""



