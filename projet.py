import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')



for i in range (20):
	plt.imshow(train_data['X'][:, :, :, i])
	plt.show()


def changeContraste(index):
	image = train_data['X'][:,:,:,index]
	

