import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

train_data = loadmat('train_32x32.mat')
train_data2 = loadmat('train_32x32.mat')
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

def Binarisation():
	#Pour chaque couleur
	for i in range(3):
		#Pour chaque image
		for j in range(lentrain):
			print("Couleur ", i ,"image ",j)
			#Initialisation de l'histogramme 
			h = [0]*256
			#Remplissage de l'histogramme
			for u in range(n):
				for v in range(n):
					h[(train_data['X'][:, :, i, j])[u][v]]+=1
			#Couleur moyenne 
			numerateur = 0
			denominateur = 0
			for k in range(len(h)):
				numerateur+=k*h[k]
				denominateur+=h[k]
			valmoy = (int)(numerateur/denominateur)
			#Binariasation
			for a in range(n):
				for b in range(n):
					if((train_data['X'][a, b, i, j])>valmoy):
						(train_data['X'][a, b, i, j]) = 255
					else:
						(train_data['X'][a, b, i, j]) = 0

Binarisation()

savemat('train_32x32_cleaned2.mat',train_data)
data_cleaned = loadmat('train_32x32_cleaned.mat')

for i in range(10):
	plt.imshow(train_data2['X'][:,:,:,i])
	plt.show()
	plt.imshow(data_cleaned['X'][:,:,:,i])
	plt.show()
