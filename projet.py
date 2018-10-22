import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

#Images de taille n*n
n = 32

"""for i in range (20):
	plt.imshow(train_data['X'][:, :, :, i])
	plt.show()


def changeContraste(index):
	compteur = 0
	#pour chaque couleur
	for i in range (3):
		#pour chaque ligne
		for j in range(n):
			#pour chaque colonne
			for k in range(n):
				compteur+=1
				if(train_data['X'][:,:,i,index][j][k]<30):
					train_data['X'][:,:,i,index][j][k]=0
			
					
	print(compteur)		
	plt.imshow(train_data['X'][:,:,i,index])
	plt.show()

def Contraste(image):
	image[image>155] = 255
	image[image<20] = 0

	plt.imshow(image[:,:,:,3])
	plt.show()

#Contraste(train_data['X'])


resultat = np.sqrt(((train_data['X'][:,:,:,0] - train_data['X'][:,:,:,1])**2).sum(-1))
print(resultat)
"""
resultat = (train_data['X'][:,:,:,0]).flatten()
print(len(resultat))