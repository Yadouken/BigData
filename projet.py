import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

image_idx = 15
print('Label:', train_data['y'][image_idx])
plt.imshow(train_data['X'][:, :, :, image_idx])
plt.show()