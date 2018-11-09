import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from collections import defaultdict
from math import sqrt

#Images de taille n*n
n = 32
lentrain = 73257
lentest = 26032

def generateTrainingData():
	train_data = loadmat("train_32x32_cleaned.mat")
	test_data = loadmat("test_32x32_cleaned.mat")
	return train_data, test_data

'''Calcul de la distance euclidienne entre deux vecteurs RGB 
	renvoie le résultat'''
def distanceEuclidienne(X,Y):
	#Flatten : applati le vecteur de taille (32,32,3) 
	#devient une liste
	a = X.flatten()
	b = Y.flatten()
	c = b-a
	return sqrt(np.dot(c,c))

#Input : image dont on veut connaitre la classe
#Distance euclidienne avec chaque représentants de classe
#Retourne la liste des distances rangés dans l'ordre
#L[i] = distance entre image et representant classe i (i de 0 à 9) 

def LDistances(image, representants):
	L = []
	for i in range(len(representants)):
		resultat = distanceEuclidienne(image, representants[i])
		L.append(resultat)
	return L

def DeterminerClasse(image, representants):
	#Récupération des distances entre chaque représentant
	L = LDistances(image, representants)
	classe = L.index(min(L))
	return classe +1

def showImageClass(data):
	plt.imshow(data[:, :, :])
	plt.show()
	
#Liste des représentants pour chaque classe 
#representants[i] = vecteur RGB représentant la classe i
def calculBarycentre(data):
	representants = dict()
	for i in range(1, 11):
		representants[i-1] = np.average(data[i], axis=0)
	return representants

def splitData(data):
	classes = defaultdict(list)

	for i in range(0, lentrain):
		classes[data["y"][i][0]].append(data["X"][:, :, :, i])

	return classes

if __name__=="__main__":
	train_data, test_data = generateTrainingData()
	orderedData = splitData(train_data)
	representants = calculBarycentre(orderedData)
	#print(representants[0])
	#print(representants[0].flatten())
	#print(DeterminerClasse(train_data["X"][:, :, :, 6]))
	#plt.imshow(train_data["X"][:, :, :, 6])
	#plt.show()
	compteur = 0
	for i in range(lentest):
		print('Image ',i)
		if(DeterminerClasse(test_data["X"][:, :, :, i], representants)==test_data["y"][i][0]):
			compteur+=1
	res = (compteur/lentest)*100
	print("Pourcentage d'images bonne classe : ", res)

