#!/usr/bin/env python
##########################################
#
# TwitterBot.py: A Simple implementation of a LSTM network to generate 140 character tweets with Keras 
#                trained on a collection of tweets with various Machine Learning hashtags.
#  
#
# Author: Cosimo Iaia <cosimo.iaia@gmail.com>
# Date: 10/11/2018
#
# This file is distribuited under the terms of GNU General Public
#
########################################

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import LambdaCallback
from keras.preprocessing import sequence
import random
import numpy as np

import argparse

FLAGS = None
char_idx = None
idx_char = None
chars= None

# fix random seed for reproducibility
np.random.seed(24)


def string_to_semi_redundant_sequences(text, seq_maxlen=128, redun_step=3, char_idx=None):
    print("Vectorizing text...")

    chars = sorted(list(set(text)))

    if char_idx is None:  
      char_idx = {c: i for i, c in enumerate(sorted(chars))}
      idx_char = {i: c for i, c in enumerate(sorted(chars))}

    len_chars=len(char_idx)

    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_maxlen, redun_step):
        sequences.append(text[i: i + seq_maxlen])
        next_chars.append(text[i + seq_maxlen])

    X = np.zeros((len(sequences), seq_maxlen, len_chars), dtype=np.bool)
    Y = np.zeros((len(sequences), len_chars), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_idx[char]] = 1
        Y[i, char_idx[next_chars[i]]] = 1
#        print(".")

    print("Text total length: {:,}".format(len(text)))
    print("Distinct chars   : {:,}".format(len_chars))
    print("Total sequences  : {:,}".format(len(sequences)))

    return X, Y, char_idx
  


def create_model(maxlen, input_length):
    model = Sequential()
    # if you are training on a GPU with cuda, cuDNNLSTM are much much faster.
    model.add(CuDNNLSTM(512,input_shape=(maxlen, input_length)))
    model.add(Dropout(0.2))
    model.add(Dense(input_length))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model
    
def gen(model,sentence,maxlen,char_idx,reslen):
    chars = sorted(list(set(text)))
    idx_char = {i: c for i, c in enumerate(sorted(chars))}

    input_length = len(char_idx)
    generated=''
   
    x = np.zeros((1, maxlen, input_length), dtype=np.bool)
    for t, char in enumerate(sentence):
        x[0, t, char_idx[char]] = 1.
    
    for i in range(reslen):
        preds = np.argmax(model.predict(x, verbose=0))
        next_char = idx_char[preds]

        generated += next_char
        act = np.zeros((1, 1, input_length), dtype=np.bool)
        act[0,0,preds] = 1
        x = np.concatenate((x[:,1:,:], act), axis=1)
        
    print(sentence+generated)
        

maxlen = 140
path = 'ML-tweets.txt'
with open(path) as f:
  text = f.read()
  
X, Y, char_idx = string_to_semi_redundant_sequences(text, seq_maxlen=maxlen)

model = create_model(maxlen, len(char_idx))
model.fit(X, Y, epochs=50, batch_size=512)
model.save('test_lstm-256lenX50ep.h5')
gen(model, "What ", maxlen, char_idx,140)

