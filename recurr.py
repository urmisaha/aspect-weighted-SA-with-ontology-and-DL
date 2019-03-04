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
from keras.layers import Activation, Conv2D, Flatten
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

os.environ['KERAS_BACKEND'] = 'tensorflow'

X_train=[]
y_train=[]
X_test=[]
y_test=[]

tokenizer_train = Tokenizer()
tokenizer_test = Tokenizer()

# file_train = open('laptop-single-sentence-train.txt', 'r')
file_train = open('laptop-full-review-train.txt', 'r')
# file_test = open('laptop-full-review-test.txt', 'r')
file_test = open('laptop-full-review-test.txt', 'r')
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

print ("Fitting on text_train...\n")
tokenizer_train.fit_on_texts(text_train)
print ("Fitting on text_test...\n")
tokenizer_test.fit_on_texts(text_test)
print ("text_train to sequence..")
X_train = tokenizer_train.texts_to_sequences(text_train)
print ("text_test to sequence..")
X_test = tokenizer_test.texts_to_sequences(text_test)
# print ("X_train>>>>>>>>>>>>>>>>>>>\n", X_train[0:5])
# print ("X_test>>>>>>>>>>>>>>>>>>>\n", X_test[0:5])


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
model.save('recursive_model.h5')
predictions = model.predict(X_test, batch_size=64, verbose=0, steps=None)
# print ("================================predictions start===================================")
# print (predictions)
# print ("=================================predictions end====================================")

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", (scores[1]*100))