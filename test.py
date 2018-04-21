import numpy as np


labels = ["banana","bulldozer","chair","eyeglasses","flashlight","foot","hand","harp","hat","keyboard","laptop","nose","parrot","penguin","pig","skyscraper","snowman","spider","trombone","violin"]
print(len(labels))
test_labels = [0,1,2,3]

with open('kmeans_submission.csv', 'w') as f:
    f.write("ID,CATEGORY")
    f.write('\n')
    for i in range(len(test_labels)):
        f.write(str(i))
        f.write(',')
        f.write(labels[test_labels[i]])
        f.write('\n')