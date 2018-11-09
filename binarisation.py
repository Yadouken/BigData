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

def Binarisation():
	#Pour chaque couleur
	for i in range(3):
		#Pour chaque image
		for j in range(lentest):
			print("Couleur ", i ,"image ",j)
			#Initialisation de l'histogramme 
			h = [0]*256
			#Remplissage de l'histogramme
			for u in range(n):
				for v in range(n):
					h[(test_data['X'][:, :, i, j])[u][v]]+=1
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
					if((test_data['X'][a, b, i, j])>valmoy):
						(test_data['X'][a, b, i, j]) = 255
					else:
						(test_data['X'][a, b, i, j]) = 0

Binarisation()

savemat('test_32x32_cleaned2.mat',test_data)
