import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

#Images de taille n*n
n = 32
#Longeurs des données d'entraînement et de test
lentrain = 73256
lentest = 26031

def sump(l,r):
	res = 0
	for i in range(r):
		res+= l[i]
	return res

def Egalisation():
	#Pour chaque couleur
	for i in range(3):
		#Pour chaque image
		for j in range(10):
			#Initialisation de l'histogramme 
			h = [0]*256
			#Remplissage de l'histogramme
			for u in range(n):
				for v in range(n):
					h[(train_data['X'][:, :, i, j])[u][v]]+=1
			#Tableau du taux de présence 
			p = [0]*256
			#Remplissage du tableau de taux de présence
			for r in range(len(p)):
				p[r] = h[r]/(n*n)
			#Egalisation
			for a in range(n):
				for b in range(n):
					val = 255*sump(p,(train_data['X'][:, :, i, j])[a][b])
					(train_data['X'][:, :, i, j])[a][b] = int(val)


Egalisation()
"""train_data['X'][train_data['X']>200] = 255
train_data['X'][train_data['X']<50] = 0
"""
savemat('train_32x32_cleaned.mat',train_data)
data_cleaned = loadmat('train_32x32_cleaned.mat')

for i in range(10):
	plt.imshow(data_cleaned['X'][:,:,:,i])
	plt.show()
