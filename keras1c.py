from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
from keras.utils.np_utils import to_categorical


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

train_labels = []
for i in range(len(train)):
    train_labels.append(int(i/5000))

train_labels = to_categorical(train_labels, num_classes=20)

model = Sequential()
model.add(Dense(500, input_dim=784, activation='sigmoid'))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, train_labels, epochs=5, batch_size=4, verbose=1)

scores = model.evaluate(train, train_labels)

print("train acc = ",scores[1]*100)
# ('train acc = ', 56.837)
