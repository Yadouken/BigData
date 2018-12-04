import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = loadmat('train_32x32_pretraitement.mat')
test_data = loadmat('test_32x32_pretraitement.mat')

lentrain = 1000
lentest = 1000

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

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train) 
y_pred = knn.predict(X_test)
for i in range (100):
	print((Y_test[i],y_pred[i]))
print(accuracy_score(Y_test, y_pred)*100)
