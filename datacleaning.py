import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

#Images de taille n*n
n = 32

def Contraste(image):
	image[image>155] = 255
	image[image<20] = 0

	
Contraste(train_data['X'])
 
plt.imshow(train_data['X'][:, :, :, 1])
plt.show()