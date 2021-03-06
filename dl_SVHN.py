import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
torch.manual_seed(0)

class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(1250, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.log_softmax(self.fc2(x), dim=1)

            return x

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.shape[0], -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 32*16*3)
        self.fc2 = nn.Linear(32*16*3, 16*16*3)
        self.fc3 = nn.Linear(16*16*3, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.contiguous().view(x.shape[0], -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

if __name__ == '__main__':

    # Load the dataset
    train_data = loadmat('train_32x32_pretraitement.mat')
    test_data = loadmat('test_32x32_pretraitement.mat')

    trainsize = 5000
    testsize = 1000

    train_label = train_data['y'][:trainsize]
    train_label = np.where(train_label==10, 0, train_label)
    train_label = torch.from_numpy(train_label.astype('int')).squeeze(1)
    train_label = train_label.type(torch.LongTensor)
    train_data = torch.from_numpy(train_data['X'].astype('float32')).permute(3, 2, 0, 1)[:trainsize]

    test_label = test_data['y'][:testsize]
    test_label = np.where(test_label==10, 0, test_label)
    test_label = torch.from_numpy(test_label.astype('int')).squeeze(1)
    test_label = test_label.type(torch.LongTensor)
    test_data = torch.from_numpy(test_data['X'].astype('float32')).permute(3, 2, 0, 1)[:testsize]

    # Hyperparameters
    epoch_nbr = 10
    batch_size = 10
    learning_rate = 0.001

    net = CNN()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    for e in range(epoch_nbr):
        print("Epoch", e)
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(train_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_train, 1)
            loss = F.nll_loss(predictions_train, train_label[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update
        

        predictions_train = net(test_data[0:testsize]) 
        _, class_predicted = torch.max(predictions_train, 1)
        succes = 0

        for i in range(0,len(class_predicted)):
            if(class_predicted[i]==test_label[i]):
                succes+=1

        print("Score test : " + str(succes/testsize*100) +"%" )

        predictions_train = net(train_data[0:trainsize]) 
        _, class_predicted = torch.max(predictions_train, 1)
        succes = 0

        for i in range(0,len(class_predicted)):
            if(class_predicted[i]==train_label[i]):
                succes+=1

        print("Score train : " + str(succes/trainsize*100) + "%")

        
