import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
import pickle
import pandas
from gensim.models import Word2Vec
from sklearn.utils import resample


dataframe = pandas.read_csv("train.csv", names=['sentence', 'sentiment'])
print(dataframe[0:5])
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

dataset = dataframe.values
X_train = dataset[0:,0]
Y_train = dataset[0:,1].astype(int)

print('Y_train:')
print(set(Y_train))

i=0
f_train = open('train_feature_vector.txt', 'w')
for sentence, label in zip(X_train, Y_train):
    word_embeddings = []
    sent_emb = Word2Vec(sentence, min_count=1, size=200)
    for word in sent_emb.wv.vocab:
        word_embeddings.append(sent_emb[word])
    emb_val = np.sum(word_embeddings, axis=0)/len(sent_emb.wv.vocab)
    f_train.write(str(Y_train[i]) + ' ')
    i = i + 1
    for val in emb_val:
        f_train.write(str(val) + ' ')
    f_train.write('\n')

f_train.close()

dataframe = pandas.read_csv("test.csv")
dataset = dataframe.values
X_test = dataset[0:,0]
Y_test = dataset[0:,1].astype(int)

print('Y_test:')
print(set(Y_test))

i=0
f_test = open('test_feature_vector.txt', 'w')
for sentence, label in zip(X_test, Y_test):
    word_embeddings = []
    sent_emb = Word2Vec(sentence, min_count=1, size=200)
    for word in sent_emb.wv.vocab:
        word_embeddings.append(sent_emb[word])
    emb_val = np.sum(word_embeddings, axis=0)/len(sent_emb.wv.vocab)
    f_test.write(str(Y_test[i]) + ' ')
    i = i + 1
    for val in emb_val:
        f_test.write(str(val) + ' ')
    f_test.write('\n')

f_test.close()

f_train = open('train_feature_vector.txt', 'r')
f_test = open('test_feature_vector.txt', 'r')

data_train = np.loadtxt(f_train)
data_test = np.loadtxt(f_test)
print(data_train.shape)
print(data_test.shape)

X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC(verbose=2)
clf = GridSearchCV(svr, parameters, verbose=2)
clf.fit(X_train, y_train)

pickle.dump(clf, open('svm.model.pkl', 'wb'))

predicted_labels = clf.predict(X_test)

print("predicted_labels: ")
print(predicted_labels)

print("precision_recall_fscore_support: ")
print(precision_recall_fscore_support(y_test, predicted_labels, average=None))
with open('predicted_labels.txt', 'w+') as f:
    for i in range(len(predicted_labels)):
        f.write(str(predicted_labels[i]))
        f.write('\n')

