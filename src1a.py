import numpy as np
from sklearn.cluster import KMeans

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

kmeans = KMeans(n_clusters=20, n_init=10, max_iter=30)

kmeans.fit(train)
print("fit done")

clusters = kmeans.labels_
labels_freq = np.zeros((20, 20))
for i in range(len(train)):
    labels_freq[clusters[i]][int(i / 5000)] += 1

cluster_label = np.argmax(labels_freq, axis=1)

# Train Accuracy
# ('train acc = ', 34.428) maxiter = 300
train_acc = 0
for i in range(len(train)):
    if cluster_label[clusters[i]] == int(i / 5000):
        train_acc += 1
print("train acc = ", train_acc * 100.0 / len(train))

# Test accuracy
# ('test acc = ', 36.317) maxiter = 300
test_labels = kmeans.predict(test)
labels = ["banana", "bulldozer", "chair", "eyeglasses", "flashlight", "foot", "hand", "harp", "hat", "keyboard",
          "laptop", "nose", "parrot", "penguin", "pig", "skyscraper", "snowman", "spider", "trombone", "violin"]
with open('kmeans_submission.csv', 'w') as f:
    f.write("ID,CATEGORY")
    f.write('\n')
    for i in range(len(test_labels)):
        f.write(str(i))
        f.write(',')
        f.write(labels[cluster_label[test_labels[i]]])
        f.write('\n')

# max iter  train   test
# 10    34.089  33.587
# 20    34.698  34.977
# 30    35.779  35.535
# 40    35.454  35.112
# 50    34.565  34.190