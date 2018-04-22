import numpy as np

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

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

train = np.append(train_banana,train_bulldozer,axis=0)
train = np.append(train,train_chair,axis=0)
train = np.append(train,train_eyeglasses,axis=0)
train = np.append(train,train_flashlight,axis=0)
train = np.append(train,train_foot,axis=0)
train = np.append(train,train_hand,axis=0)
train = np.append(train,train_harp,axis=0)
train = np.append(train,train_hat,axis=0)
train = np.append(train,train_keyboard,axis=0)
train = np.append(train,train_laptop,axis=0)
train = np.append(train,train_nose,axis=0)
train = np.append(train,train_parrot,axis=0)
train = np.append(train,train_penguin,axis=0)
train = np.append(train,train_pig,axis=0)
train = np.append(train,train_skyscraper,axis=0)
train = np.append(train,train_snowman,axis=0)
train = np.append(train,train_spider,axis=0)
train = np.append(train,train_trombone,axis=0)
train = np.append(train,train_violin,axis=0)

test = np.load('test/test.npy')

# pca = PCA(n_components=50, copy=False)

# reduced = pca.fit_transform(train)
# reduced = normalize(reduced, axis=1)
# np.save("pca_norm_axis1",reduced)

# test_reduced = pca.transform(test)
# test_reduced = normalize(test_reduced, axis=1)
# np.save("test_pca_norm_axis1",test_reduced)

reduced = np.load("pca_norm_axis1.npy")
test_reduced = np.load("test_pca_norm_axis1.npy")


train_labels = []
for i in range(len(train)):
    train_labels.append(int(i/5000))


svm = SVC(kernel='linear',decision_function_shape='ovo',verbose=True)
svm.fit(reduced,train_labels)

print("Train acc = ", svm.score(reduced,train_labels))
# ('Train acc = ', 0.47745)
# ('Train acc = ', 0.65346)


test_labels = svm.predict(test_reduced)
labels = ["banana", "bulldozer", "chair", "eyeglasses", "flashlight", "foot", "hand", "harp", "hat", "keyboard",
          "laptop", "nose", "parrot", "penguin", "pig", "skyscraper", "snowman", "spider", "trombone", "violin"]
with open('svm_submission.csv', 'w') as f:
    f.write("ID,CATEGORY")
    f.write('\n')
    for i in range(len(test_labels)):
        f.write(str(i))
        f.write(',')
        f.write(labels[test_labels[i]])
        f.write('\n')

# test 48.635
# test 65.259

# 19:06