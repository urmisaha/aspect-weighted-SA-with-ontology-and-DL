import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, confusion_matrix, \
        classification_report, accuracy_score, f1_score, \
        precision_recall_fscore_support

f_test = open('test_feature_vector.txt', 'r')
data_test = np.loadtxt(f_test)
print(data_test.shape)

X_test = data_test[:, 1:]
y_test = data_test[:, 0]
print(X_test.shape)
print(y_test.shape)

clf = pickle.load(open('svm.model.pkl', 'rb'))

predicted_labels = clf.predict(X_test)
print(precision_recall_fscore_support(y_test,predicted_labels,average=None)) 
print('Accuracy:', accuracy_score(y_test, predicted_labels))
print('F1 score:', f1_score(y_test, predicted_labels))
print('Recall:', recall_score(y_test, predicted_labels))
print('Precision:', precision_score(y_test, predicted_labels))
print('clasification report:\n', classification_report(y_test,predicted_labels))
print('confussion matrix:\n',confusion_matrix(y_test, predicted_labels)) 
