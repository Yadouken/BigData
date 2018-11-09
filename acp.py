import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

lentrain = 73257
lentest = 26032

train_flattened = [[] for x in range(lentrain)]
test_flattened =  [[] for y in range(lentest)]
for i in range(lentrain):
	train_flattened[i] = (train_data['X'][:, :, :, i]).flatten()
for j in range(lentest):
	test_flattened[j] = (test_data['X'][:, :, :, j]).flatten()

print(test_flattened[0])
print(len(test_flattened[0]))
#n_components maximal = 32*32*3
#Les dernières composantes sont que du bruit (expliquent mal les données)
#Essayer les premières (augemnter n_components) et recueillir les infos dans le rapport
#Paramètres à donner à pca : Liste d'images applaties (32*32*3 via un flatten)
pca = PCA(n_components=2)
test_pca = pca.fit_transform(test_flattened)  

print(test_pca[0])
print(len(test_pca[0]))