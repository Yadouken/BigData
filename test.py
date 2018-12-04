import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb

#Images de taille n*n
n = 32
#Longeurs des données d'entraînement et de test
lentrain = 73257
lentest = 26032

def rgbToGray(data):
	for i in range(lentrain):
		print("rgb ",i)
		for x in range(0, n):
			for y in range(0, n):
				for j in range(0, 3):
					data["X"][x, y, j, i] = np.average(data["X"][x, y, :, i])
	return data

def Pretraitement(train_data):
	#Contraste
	for i in range(lentrain):
		print("Image ",i)
		for j in range(3):
			#Détermination des valeurs minimales et maximales des pixels
			pixelMin = train_data['X'][0, 0, 0, i];
			pixelMax = train_data['X'][0, 0, 0, i];
			for u in range(n):
				for v in range(n):
					if(train_data['X'][u, v, j, i]>pixelMax):
						pixelMax = train_data['X'][u, v, j, i]
					elif(train_data['X'][u, v, j, i]<pixelMin):
						pixelMin = train_data['X'][u, v, j, i]
			scale = float(float(255.0)/float((pixelMax - pixelMin)))
			for u in range(n):
				for v in range(n):
					train_data['X'][u, v, j, i] = (train_data['X'][u, v, j, i]-pixelMin)*scale
	
	#Passage en gris et ajustement du background
	train_data = rgbToGray(train_data)
	for h in range(lentrain):
		print("Inversion ", h)
		compteurSombre = np.count_nonzero((train_data['X'][:, :, 0, h].flatten())<=127)
		if(compteurSombre>=512):
			for w in range(3):
				for x in range(n):
					for y in range(n):
						train_data['X'][x, y, w, h] = 255-train_data['X'][x, y, w, h] 

if __name__=="__main__":
	train_data = loadmat('train_32x32.mat')
	test_data = loadmat('test_32x32.mat')

	Pretraitement(train_data)
	savemat('train_32x32_pretraitement.mat',train_data)
	data_cleaned = loadmat('train_32x32_pretraitement.mat')
	
	for i in range(10):
		plt.imshow(data_cleaned['X'][:,:,:,i])
		plt.show()
