import numpy
import pandas
import os
import re
import itertools
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from metrics import *
from gensim.models import Word2Vec
# from PreprocessingText import *


os.environ['KERAS_BACKEND'] = 'theano'

X_train=[]
y_train=[]
X_test=[]
y_test=[]

tokenizer_train = Tokenizer()
tokenizer_test = Tokenizer()

file_train = open('dataset/laptop-full-review-train.txt', 'r')
file_test = open('dataset/laptop-full-review-test.txt', 'r')
text_train = []
text_test = []

# for line_train, line_test in itertools.zip_longest(file_train, file_test):
for idx,line in enumerate(file_train):
	y_train.append(int(line.split("   ")[1]))
	text_train.append(line.split("   ")[0])
for idx,line in enumerate(file_test):
	y_test.append(int(line.split("   ")[1]))
	text_test.append(line.split("   ")[0])
	
file_train.close()
file_test.close()

# print ("Fitting on text_train...\n")
# tokenizer_train.fit_on_texts(text_train)
# print (tokenizer_train.fit_on_texts(text_train))
# print ("Fitting on text_test...\n")
# tokenizer_test.fit_on_texts(text_test)
# print ("text_train to matrix..")
# X_train = tokenizer_train.texts_to_matrix(text_train, mode='count')
# print ("text_test to matrix..")
# X_test = tokenizer_test.texts_to_matrix(text_test, mode='count')
# print ("X_train>>>>>>>>>>>>>>>>>>>\n", X_train[0:5])
# print ("X_test>>>>>>>>>>>>>>>>>>>\n", X_test[0:5])

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
# nb_words = 500
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# print (numpy.array(X_train).shape, numpy.array(y_train).shape, numpy.array(X_test).shape, numpy.array(y_test).shape)
# print ("Xtrain and ytrain:::::::::: \n", numpy.array(X_train)[0], numpy.array(y_train)[0])
# print ("\nXtest and ytest:::::::::: \n", numpy.array(X_test)[0], numpy.array(y_test)[0])

# reviews = [[word.lower() for word in text.split()] for text in text_train]
X_train_reviews = process_reviews(text_train)
X_test_reviews = process_reviews(text_test)
# texts = [text_to_word_sequence(text.flatten(), filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')for text in texts]
# print (texts[0:2])

embedding_model_train = Word2Vec(X_train_reviews, min_count=1)
embedding_model_test = Word2Vec(X_test_reviews, min_count=1)
# print (embedding_model)
word_vectors_train = embedding_model_train.wv

MAX_NB_WORDS = len(word_vectors_train.vocab)
MAX_SEQUENCE_LENGTH = 10

word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}
# print (word_index)
# print("Number of word vectors: {}".format(len(word_vectors_train.vocab)))

train_sequences = [[word_index.get(t, 0) for t in review] for review in X_train_reviews]
test_sequences = [[word_index.get(t, 0)  for t in review] for review in X_test_reviews]

# print (train_sequences[0:5])

X_train = sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="pre")
X_test = sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="pre")

print (X_train[0:5])

WV_DIM = 20
nb_words = min(MAX_NB_WORDS, len(word_vectors_train.vocab))
# we initialize the matrix with random numbers
wv_matrix = (numpy.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
c=0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors_train[word]
        # if c == 0:
	        # print (embedding_vector)
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
        # c = 1
    except:
        pass

print (wv_matrix.shape)
# print (wv_matrix)
# print (words[0])
# print (embedding_model[words[0]])
# embedded_text = list(embedding_model.wv)
# print(embedded_text[0])
# print(len(embedded_text))

# truncate and pad input sequences
# max_review_length = 500
# X_train = sequence.pad_sequences(embedded_text, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
# WV_DIM = 32
model = Sequential()
model.add(Embedding(nb_words, WV_DIM, weights=[wv_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True))
model.add(Bidirectional(LSTM(5)))

# BiLSTM
# model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Attention
# model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f2_score, precision, recall])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=256)
model.save('recurrent_model.h5')

predictions = model.predict(X_test, batch_size=256, verbose=0, steps=None)
# print ("================================predictions start===================================")
# print (predictions[0:20])
# print ("=================================predictions end====================================")

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
# print("Scores: ", scores)
# print("Accuracy: ", (scores[1]*100))
# print("F2: ", (scores[2]*100))
