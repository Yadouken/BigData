import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from collections import defaultdict

#Images de taille n*n
n = 32

def generateTrainingData():
	train_data = loadmat("train_32x32.mat")
	test_data = loadmat("test_32x32.mat")
	return train_data, test_data

#Calcul de la distance euclidienne entre deux vecteurs RGB 
#renvoit le résultat
def distanceEuclidienne(X,Y):
	#Flatten : applati le vecteur de taille (32,32,3) 
	#devient une liste
	a = X.flatten()
	b = Y.flatten()
	res = np.dot(a,b)
	return res

#Input : image dont on veut connaitre la classe
#Distance euclidienne avec chaque représentants de classe
#Retourne la liste des distances rangés dans l'ordre
#L[i] = distance entre image et representant classe i (i de 0 à 9) 
def LDistances(image):
	L = []
	for i in range(len(representants)):
		resultat = distanceEuclidienne(image, representants[i])
		L.append(resultat)
	return L

def DeterminerClasse(image):
	#Récupération des distances entre chaque représentant
	L = LDistances(image)
	return L.index(min(L))

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

	for i in range(0, len(data["y"])):
		classes[data["y"][i][0]].append(data["X"][:, :, :, i])

	return classes

if __name__=="__main__":
	train_data, test_data = generateTrainingData()
	orderedData = splitData(train_data)
	representants = calculBarycentre(orderedData)
	print(DeterminerClasse(test_data["X"][:, :, :, 25229]))
	plt.imshow(test_data["X"][:, :, :, 25229])
	plt.show()
