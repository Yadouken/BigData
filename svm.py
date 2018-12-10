import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

lentrain = 10000
lentest = 2000

X_train = [[] for x in range(lentrain)]
X_test =  [[] for y in range(lentest)]
for i in range(lentrain):
	X_train[i] = (train_data['X'][:, :, :, i]).flatten()
for j in range(lentest):
	X_test[j] = (test_data['X'][:, :, :, j]).flatten()

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

Y_train = np.zeros(lentrain)
Y_test = np.zeros(lentest)
for u in range(lentrain):
	Y_train[u] = train_data['y'][u][0]
for v in range(lentest):
	Y_test[v] = test_data['y'][v][0]

svclassifier = SVC(C=0.9, cache_size=200, class_weight=None, coef0 =0.0, decision_function_shape='ovo', gamma=0.2,kernel="poly", max_iter=-1, probability = False, random_state = None, shrinking=True, tol=0.001,verbose = False)  

print("Entrainement")
svclassifier.fit(X_train, Y_train) 

print("Prediction")
y_pred = svclassifier.predict(X_test)

accuracy = svclassifier.score(X_test, Y_test)
print("Accuracy : ", accuracy*100)

# print("Resultats")
#print(confusion_matrix(Y_test,y_pred))  
#print(classification_report(Y_test,y_pred))  