import string
import numpy as np
import random
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from DTTmath import Data

class TextGenerator(Data):

    def __init__(self, init_sent_path, training_text_path, no_lines, hidden_size, batch_size, epochs):
        self.init_sent_path = init_sent_path
        self.training_text_path = training_text_path
        self.no_lines = no_lines
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def open_text(path):
        text_file_open = open(path, "r")
        text = text_file_open.read()
        return text
        
    @staticmethod
    def clean_text(text_dirty):
        tokens = text_dirty.split()
        table = str.maketrans("","", string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word.lower() for word in tokens]
        return tokens
    
    @staticmethod
    def generate_lines(length, tokens):
        lines = []
        for i in range(length, len(tokens)):
            seq = tokens[i-length:i]
            line = ' '.join(seq)
            lines.append(line)
            if i > 200000:
                break
        return lines
    
    @staticmethod
    def get_bigrams(text_tokens):
        list_bigrams = []
        for line in text_tokens:
            one_sent = []
            for i in range(0, len(line)-1):
                one_sent.append(line[i]+" "+line[i+1])
            list_bigrams.append(one_sent)
        return list_bigrams
    

    
#Przygotowanie zdania inicjującego (seed):
    def pick_init_sent(self):
        init_sent_open = open(self.init_sent_path, "r")
        init_sent = init_sent_open.read()
        text = init_sent.split("\n")
        if text[-1] == "":
            text.pop()
        random.shuffle(text)
        init_sent_picked = random.choice(text)
        seed_text = self.clean_text(init_sent_picked)
        seed_text = ' '.join(seed_text)
        return seed_text
    
#Wektoryzacja
#Wbudowany tokenizer - nie robi n-gramów potrzebnych do znaków interpunkcyjnych
    @staticmethod
    def get_vocab_built_tokenizer(tokenizer):
        vocab_size = len(tokenizer.word_index) + 1
        return vocab_size
    
    def tokenizer_built_in(self, tokens):
        lines = self.generate_lines(self.no_lines, tokens)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        sequences = tokenizer.texts_to_sequences(lines)
        sequences = np.array(sequences)
        return sequences
        
#Word2Vec with n-grams:
    @staticmethod
    def get_vocab_word2vec(pretrained_weights):
        vocab_size, embedding_size = pretrained_weights.shape
        return vocab_size
    
    def tokenizer_word2vec(self):
        text_preprocessed = []
        text = self.open_text(self.training_text_path).split("\n")
        text.pop()
        for line in text:
            text_preprocessed.append(gensim.utils.simple_preprocess(line))
        bigrams = self.get_bigrams(text_preprocessed)
        text_preprocessed.extend(bigrams)
        model_word2vec = gensim.models.Word2Vec(text_preprocessed, size=4, window=3, min_count=1, workers=30, iter=300)
        pretrained_weights = model_word2vec.wv.syn0
        return pretrained_weights
    
    @staticmethod
    def get_X_y(sequences, vocab_size):
        X, y = sequences[:, :-1], sequences[:,-1]
        seq_length = X.shape[1]
        y = to_categorical(y, num_classes = vocab_size)
        return X, y, seq_length

    def train_LSTM(self, X, y, vocab_size, embedding_size, pretrained_weights):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))
        model.add(LSTM(self.hidden_size, return_sequences=True))
        model.add(LSTM(self.hidden_size))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(vocab_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, batch_size = self.batch_size, epochs = self.epochs)
        
#Generacja tekstu
    @staticmethod
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
            """
            if (next_word == 'datasubject' and data_intro['is_date'] == False) or next_word == 'diffminmax':
                seed_text = seed_text + ' ' + next_word
                text.append(next_word + '.')
                break
            else:
            """
            seed_text = seed_text + ' ' + next_word
            text.append(next_word)
        return ' '.join(text)

    @staticmethod
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

    @staticmethod
    def capitalizeAfterDot(text_no_cap):
        text_no_cap = text_no_cap.split()
        for i in range(1, len(text_no_cap)):
            if text_no_cap[i-1].find(".") != -1:
                text_no_cap[i] = text_no_cap[i].capitalize()
        return ' '.join(text_no_cap)

    @staticmethod
    def isDateReduction(text, data_intro):
        text = text.split('.')
        return text[0]
    
    @staticmethod
    def removeDuplicates():
        pass

    @staticmethod
    def replace_type_subject(data_type, data_subject, text_generated, data_intro):
        text_generated = text_generated.replace("datatype", data_type)
        text_generated = text_generated.replace("datasubject", data_subject)
        if 'start_date' in data_intro:
            text_generated = text_generated.replace("startdate", str(data_intro['start_date']))
        if 'end_date' in data_intro:
            text_generated = text_generated.replace("enddate", str(data_intro['end_date']))      
        return text_generated
    
text = TextGenerator("initialEnglish.txt", "intro.txt", 10, 150, 256, 40)
seed_text = text.pick_init_sent()
tokens = text.clean_text(text.open_text(text.training_text_path))
seq = text.tokenizer_built_in(tokens)
weights = text.tokenizer_word2vec()


"""
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



