import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np


# Hyper Parameters 
input_size = 784
hidden_size = 500
num_classes = 1 #20
num_epochs = 5
batch_size = 100
learning_rate = 0.001

train_banana = np.load('train/banana.npy')
train_bulldozer = np.load('train/bulldozer.npy')
train_chair = np.load('train/chair.npy')
train_eyeglasses = np.load('train/eyeglasses.npy')
train_flashlight = np.load('train/flashlight.npy')
train_foot = np.load('train/foot.npy')
train_hand = np.load('train/hand.npy')
train_harp = np.load('train/harp.npy')
train_hat = np.load('train/hat.npy')
train_keyboard = np.load('train/keyboard.npy')
train_laptop = np.load('train/laptop.npy')
train_nose = np.load('train/nose.npy')
train_parrot = np.load('train/parrot.npy')
train_penguin = np.load('train/penguin.npy')
train_pig = np.load('train/pig.npy')
train_skyscraper = np.load('train/skyscraper.npy')
train_snowman = np.load('train/snowman.npy')
train_spider = np.load('train/spider.npy')
train_trombone = np.load('train/trombone.npy')
train_violin = np.load('train/violin.npy')

train = np.append(train_banana, train_bulldozer, axis=0)
train = np.append(train, train_chair, axis=0)
train = np.append(train, train_eyeglasses, axis=0)
train = np.append(train, train_flashlight, axis=0)
train = np.append(train, train_foot, axis=0)
train = np.append(train, train_hand, axis=0)
train = np.append(train, train_harp, axis=0)
train = np.append(train, train_hat, axis=0)
train = np.append(train, train_keyboard, axis=0)
train = np.append(train, train_laptop, axis=0)
train = np.append(train, train_nose, axis=0)
train = np.append(train, train_parrot, axis=0)
train = np.append(train, train_penguin, axis=0)
train = np.append(train, train_pig, axis=0)
train = np.append(train, train_skyscraper, axis=0)
train = np.append(train, train_snowman, axis=0)
train = np.append(train, train_spider, axis=0)
train = np.append(train, train_trombone, axis=0)
train = np.append(train, train_violin, axis=0)

test = np.load('test/test.npy')

# train_dataset = train

# test_dataset = test
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                           batch_size=batch_size, 
#                                           shuffle=False)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.soft = nn.Softmax()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out
    
net = Net(input_size, hidden_size, num_classes)

    
# Loss and Optimizer
criterion = nn.NLLLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

train_labels = []
for i in range(len(train)):
    train_labels.append(int(i/5000))

train_labels = np.array(train_labels)
# Train the Model
for epoch in range(num_epochs):
    # for i, (images, labels) in enumerate(train_loader):  
    i = 0
    for images in train:
        labels = train_labels[i]
        labels = np.array(labels)
        # Convert torch tensor to Variable
        images = Variable(torch.from_numpy(images).view(28*28)).float()
        labels = Variable(torch.from_numpy(labels))
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
        i += 1

# Train accuracy
train_acc = 0
i = 0
for images in train:
    labels = train_labels[i]
    images = Variable(torch.from_numpy(images).view(-1, 28*28))
    outputs = net(images)
    predicted = int(outputs.data[0]*20)
    if (predicted == labels):
        correct += 1

print('Train accuracy = ',correct / len(train_labels))
# Save the Model
torch.save(net.state_dict(), 'model.pkl')
print("saved")