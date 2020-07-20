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
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


data_intro = data_intro()

#Przygotowanie zdania inicjującego (seed):

init_sent_open = open("initialEnglish.txt", "r")
init_sent = init_sent_open.read()

def pickInitSent(text_file):
    text = text_file.split("\n")
    if text[-1] == "":
        text.pop()
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
    table = str.maketrans("","", string.punctuation)
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
    
#Wektoryzacja

#Wbudowany Tokenizer - nie robi n-gramów potrzebnych do znaków interpunkcyjnych
lines = generate_lines(9+1)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
#bigrams = ngrams(word_tokenize(text_to_train), 2)
sequences = tokenizer.texts_to_sequences(lines)
sequences = np.array(sequences)

#Word2Vec - gensim z n-gramami

def getBigrams(text_tokens):
    list_bigrams = []
    for line in text_tokens:
        one_sent = []
        for i in range(0, len(line)-1):
            one_sent.append(line[i]+" "+line[i+1])
        list_bigrams.append(one_sent)
    return list_bigrams
    
import gensim
mojtekst_preprocessed = []
mojtekst = text_to_train.split("\n")
mojtekst.pop()


for line in mojtekst:
    mojtekst_preprocessed.append(gensim.utils.simple_preprocess(line))
    
mojebigramy = getBigrams(mojtekst_preprocessed)
mojtekst_preprocessed.extend(mojebigramy)
model_word2vec = gensim.models.Word2Vec(mojtekst_preprocessed, size=4, window=3, min_count=1, workers=30, iter=300)

pretrained_weights = model_word2vec.wv.syn0
vocab_size, embedding_size = pretrained_weights.shape


X, y = sequences[:, :-1], sequences[:,-1]
seq_length = X.shape[1]
y = to_categorical(y, num_classes = vocab_size)
"""
vocab_size = len(tokenizer.word_index) + 1

model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, 
                    weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))

"""
#LSTM Model
hidden_size = 150

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size))
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
        if (next_word == 'datasubject' and data_intro['is_date'] == False) or next_word == 'diffminmax':
            seed_text = seed_text + ' ' + next_word
            text.append(next_word + '.')
            break
        else:
            seed_text = seed_text + ' ' + next_word
            text.append(next_word)
    return ' '.join(text)


def addPunctMark(text_train, text_no_punct, mark, mark2):
    stop_bigrams_dirty = []
    text_punct = []
    text_train = text_train.split()
    text_no_punct = text_no_punct.split()
    
    for i in range(0, len(text_train) - 1):
        text_train[i] = text_train[i].lower()
        text_train[i] = text_train[i].replace("_", "")
        text_train[i + 1] = text_train[i + 1].lower()
        text_train[i + 1] = text_train[i + 1].replace("_", "")
        if text_train[i + 1].find(mark2) != -1:
            text_train[i + 1]= text_train[i + 1][:-1]
        if text_train[i].find(mark) != -1:
            stop_bigrams_dirty.append(text_train[i] + " " + text_train[i+1])

    for i in range(0, len(text_no_punct) - 1):
        temp_bigram = text_no_punct[i]+mark+" "+text_no_punct[i+1]
        if temp_bigram in stop_bigrams_dirty:
            text_punct.append(text_no_punct[i] + mark)
        else:
            text_punct.append(text_no_punct[i])
    text_punct.append(text_no_punct[len(text_no_punct) - 1])
                      
    return " ".join(text_punct)


def capitalizeAfterDot(text_no_cap):
    text_no_cap = text_no_cap.split()
    for i in range(1, len(text_no_cap)):
        if text_no_cap[i-1].find(".") != -1:
            text_no_cap[i] = text_no_cap[i].capitalize()
    return ' '.join(text_no_cap)


def isDateReduction(text, data_intro):
    text = text.split('.')
    return text[0]

def removeDuplicates():
    pass

def replace_type_subject(data_type, data_subject, text_generated, data_intro):
    text_generated = text_generated.replace("datatype", data_type)
    text_generated = text_generated.replace("datasubject", data_subject)
    if 'start_date' in data_intro:
        text_generated = text_generated.replace("startdate", str(data_intro['start_date']))
    if 'end_date' in data_intro:
        text_generated = text_generated.replace("enddate", str(data_intro['end_date']))      
    return text_generated

text_generated = generate_next_seq(model, tokenizer, seq_length, seed_text, 160)
text_generated = addPunctMark(text_to_train, text_generated, ".", ",")
text_generated = addPunctMark(text_to_train, text_generated, ",", ".")
text_generated = seed_text.capitalize() + ". " + text_generated
text_generated = capitalizeAfterDot(text_generated)

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
#if text_generated != "the largest value was maxvalue and it was in totalmaxdate the smallest value was minvalue and it was in totalmindate it is noteworthy that over the whole period of time the columnname tendencyglobal after a deeper analysis it can be seen that between datefirsttendencylocal and datesecondtendencylocal the columnname is tendencylocalfirst and between datethirdtendencylocal and datefourthtendencylocal tendencylocalsecond the yoy indicator is around yoyvalue with the largest increase seen between yearfirstyoy and yearsecondyoy and the smallest between yearthirdyoy and yearfourthyoy the largest x values over the entire period are valuefirsttop valuesecondtop valuethirdtop valuextop for yearfirsttop yearsecondtop yearthirdtop and yearxtop respectively accordingly the x of the smallest values over the entire time period is valuefirstbottom valuesecondbottom valuethirdbottom valuexbottom for yearfirstbottom yearsecondbottom yearthirdbottom yearxbottom the average of columnname is averagevalue and the sum of columnname in all periods equals to sumvalue the difference between the smallest and largest value of columnname is diffminmax.":
#file1.write(seed_text + ". " + text_generated + "\n" + "\n")
file1.write(text_generated + "\n" + "\n")  
file1.close() 
"""


# 5/100 dobre - 4 + 1 - zapętlenia, nie widzi roznicy miedzy min i max
#52/100 dobre - 6 + 1 - głównie zapętlenia
#76/100 dobre - 8 + 1
#81/100 dobre - 9 + 1
#73/100 dobre - 10 + 1 - kombinowało samo jakieś bzdury



