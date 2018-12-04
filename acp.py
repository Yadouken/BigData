import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

train_data = loadmat('train_32x32_pretraitement.mat')
test_data = loadmat('test_32x32_pretraitement.mat')

lentrain = 73257
lentest = 26032

train_flattened = [[] for x in range(lentrain)]
test_flattened =  [[] for y in range(lentest)]
for i in range(lentrain):
	train_flattened[i] = (train_data['X'][:, :, :, i]).flatten()
for j in range(lentest):
	test_flattened[j] = (test_data['X'][:, :, :, j]).flatten()

#n_components maximal = 32*32*3
#Les dernières composantes sont que du bruit (expliquent mal les données)
#Essayer les premières (augemnter n_components) et recueillir les infos dans le rapport

print("Reduction de la dimension en cours")
pca = PCA(n_components=100)
train_pca = pca.fit_transform(train_flattened) 
test_pca = pca.fit_transform(test_flattened) 
print("PCA finished successfully")

np.save('train_pca100_pretraitement.npy', train_pca)
np.save('test_pca100_pretraitement.npy', test_pca)
 
print("Fichiers sauvegardés")

