from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical


# Hyper Parameters 
input_size = 28*28
hidden_size = 500
num_classes = 20
num_epochs = 5
batch_size = 32

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

train_labels = to_categorical(train_labels, num_classes=num_classes)

model = Sequential()
model.add(Dense(hidden_size, input_dim=input_size, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

scores = model.evaluate(train, train_labels)
print("train acc = ",scores[1]*100)

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# test_labels = loaded_model.predict(test)
test_labels = model.predict(test)
labels = ["banana", "bulldozer", "chair", "eyeglasses", "flashlight", "foot", "hand", "harp", "hat", "keyboard",
          "laptop", "nose", "parrot", "penguin", "pig", "skyscraper", "snowman", "spider", "trombone", "violin"]
with open('nn_submission.csv', 'w') as f:
    f.write("ID,CATEGORY")
    f.write('\n')
    for i in range(len(test_labels)):
        f.write(str(i))
        f.write(',')
        f.write(labels[np.argmax(test_labels[i])])
        f.write('\n')


# ('train acc = ', 54.946) e 1 b 32 h1000
# ('train acc = ', 61.894) e 5 b 32 h1000
# ('train acc = ', 63.814) e10 b32 h500
# ('train acc = ', 65.568) e10 b32 h784
# ('train acc = ', 65.726) e10 b32 h1000
# ('train acc = ', 62.721) e10 b32 h1500

# test 54.4 e1 b32 h1000
# test 61.305 e5 b32 h1000
# test 62.990 e10 b32 h500
# test 64.437 e10 b32 h784
# test 64.810 e10 b32 h1000
# test 61.917 e10 b32 h1500
