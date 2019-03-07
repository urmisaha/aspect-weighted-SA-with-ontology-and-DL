import numpy
import pandas
import os, re
import itertools
# from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Bidirectional, Dropout, Flatten
from keras.layers import TimeDistributed
from keras.utils import np_utils
from metrics import *
from sklearn.metrics import *
from sklearn import preprocessing
from gensim.models import Word2Vec
from PreprocessingText import *
from similarity import *
from aspect_weights_cpt import *
from sklearn.utils import resample
# from accuracy_FScore import *


os.environ['KERAS_BACKEND'] = 'theano'

tokenizer_train = Tokenizer()
tokenizer_test = Tokenizer()
 
# fix random seed for reproducibility
numpy.random.seed(7)

# Reading and storing data 
# dataframe = pandas.read_csv("train.csv", header=None)
dataframe = pandas.read_csv("train.csv", header=None, names=['sentence', 'sentiment'])
df_majority = dataframe[dataframe['sentiment']==1]
df_minority = dataframe[dataframe['sentiment']==0]

df_majority_upsampled = resample(df_majority, 
                                 replace=True,     # sample with replacement
                                 n_samples=500,   # to match majority class
                                 random_state=123) # reproducible results


df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=500,   # to match majority class
                                 random_state=123) # reproducible results

dataframe = pandas.concat([df_majority_upsampled, df_minority_upsampled])
print("dataframe.shape = ", dataframe.shape)
dataset = dataframe.values
X_train = dataset[0:,0]
Y_train = dataset[0:,1].astype(int)
Y_train = np_utils.to_categorical(Y_train)

dataframe = pandas.read_csv("test.csv", header=None)
dataset = dataframe.values
X_test = dataset[0:,0]
Y_test = dataset[0:,1].astype(int)
Y_test_nc = Y_test
Y_test = np_utils.to_categorical(Y_test)

X_train_reviews = process_reviews(X_train)
X_test_reviews = process_reviews(X_test)

WV_DIM = 200

# Word Embeddings
embedding_model_train = Word2Vec(X_train_reviews, min_count=1, size=WV_DIM)
# print (embedding_model_train)         # Word2Vec(vocab=1881, size=100, alpha=0.025)

word_vectors_train = embedding_model_train.wv
# print (word_vectors_train)            # Word2VecKeyedVectors

MAX_NB_WORDS = len(word_vectors_train.vocab)
# print("MAX_NB_WORDS:", MAX_NB_WORDS)  # 1881

nb_words = min(MAX_NB_WORDS, len(word_vectors_train.vocab))
# print("nb_words:", nb_words)          # 1881

 
# Weighted Word Embeddings 
words = list(embedding_model_train.wv.vocab)
wv_matrix = []
for word in words:
    if word in aspect_term_list:
        wv_matrix.append(embedding_model_train[word] * aspect_weights[aspect_term_mapping[word]])
    else:
        wv_matrix.append(embedding_model_train[word])

# Normalizing the weight matrix
# wv_matrix = preprocessing.normalize(wv_matrix)


# Mapping words of sentences to unique numbers  
MAX_SEQUENCE_LENGTH = 50

# creates a dictionary with index for each unique word in dataset. This index is used instead of words as input
word_index = {t[0]: i for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}

train_sequences = [[word_index.get(t, 0) for t in review] for review in X_train_reviews]
test_sequences = [[word_index.get(t, 0)  for t in review] for review in X_test_reviews]

# print ("train_sequences[0:5]:")
# print (train_sequences[0:5])

X_train = sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
X_test = sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")

# Model(LSTM) 

# Normalizing the input data
# X_train = preprocessing.normalize(X_train)
# X_test = preprocessing.normalize(X_test)


model = Sequential()
model.add(Embedding(nb_words, WV_DIM, weights=[numpy.array(wv_matrix)], input_length=MAX_SEQUENCE_LENGTH, trainable=True))
model.add(LSTM(100,input_shape=(nb_words, WV_DIM)))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # softmax/sigmoid
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_train, Y_train), validation_split=0.8, epochs=5, batch_size=256)
# model.save('model.h5')

# print("Y_test = ", Y_test)
predictions = model.predict(X_test, batch_size=256, verbose=0, steps=None)
print("predictions = ", predictions)

# predictions1 = predictions
predictions1 = []
for p in predictions:
    if p[0] < p[1]:
        predictions1.append(0)
    else:
        predictions1.append(1)

print("predictions1 = ", predictions1)
# Final evaluation of the model
# scores = model.evaluate(X_test, Y_test, verbose=0)
# print("scores: ", scores)


# Scores
print("Scores calculated from sklearn::")
print("accuracy_score: ", accuracy_score(Y_test_nc, predictions1))
print("precision_score: ", precision_score(Y_test_nc, predictions1))
print("recall_score: ", recall_score(Y_test_nc, predictions1))
# print("f1_score: ", f1_score(Y_test_nc, predictions))
print("\nClassification Report:")
print(classification_report(Y_test_nc, predictions1))





################## Extra ##################

# print("word_vectors_train.word_vec(food): ", word_vectors_train.word_vec('food', use_norm=True))
# print ("predictions[0] === ", predictions[0])
# prediction_probs = model.predict_proba(X_test)
# print ("prediction_probs[0] ==== ", prediction_probs[0])
# print (model.get_weights())
# print (model.layers)
# print ("================================predictions start===================================")
# print (predictions[0:20])
# print ("=================================predictions end====================================")

# model.add(Dense(250, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(100, activation='relu'))