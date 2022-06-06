import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()

X, y = iris.data, iris.target

print("\n iris.data = ", X)
print("\n iris.target = ", y)

X1, y1 = datasets.make_classification(
    n_samples = 100,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_repeated = 0,
    random_state = 3
)

print("\n X1 = ", X1)
print("\n y1 = ", y1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=1234)

#print(X_train.shape) 
## 120 - number of samples
## 4 - number of features for each sample

## print first sample:
#print(X_train[0]) # 4 features in it

## trainins labels :

#print(y_train.shape) #1d row vector of size 120

#print(y_train)

#plt.figure()
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20) # plot only first 2 features
#plt.show()


## list a:
#a = [1, 1, 1, 2, 2, 3, 4, 5, 6,]
#from collections import Counter
#most_common = Counter(a).most_common(1)
## most_common = Counter(a).most_common(2)
## [(1, 3), (2, 2)] <- "1" -3 times, "2" - 2 times
#print(most_common[0][0])
## 1
## [(1, 3)] <- most common items - value = 1, times = 3

from knn import KNN

clf = KNN(n_neighbors=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

#x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
#y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1

#xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

#Z1 = clf.predict(xx)
#Z2 = clf.predict(X_test)

print("\n X_train = ", X_train)
print("\n y_train = ", y_train)
print("\n predictions = ", predictions)

#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20) 
#plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               color=cmap(idx), marker=markers[idx], label=cl)
plt.xlabel('x - label')
plt.ylabel('y - label')
plt.title('Iris - data')    
plt.show()

clf.score(y_test, X_test)

sc = StandardScaler()
sc.fit(X_train1)
X_train_std = sc.transform(X_train1)
X_test_std = sc.transform(X_test1)

markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X1[y1 == cl, 0], y=X1[y1 == cl, 1],
               color=cmap(idx), marker=markers[idx], label=cl)
plt.xlabel('x - label')
plt.ylabel('y - label')
plt.title('random data')   
plt.show()

#calculate accuracy:

acc = np.sum(predictions == y_test) / len(y_test)




#print(acc)

# --------------------------------
#import numpy as np
#from matplotlib import pyplot as plt
#from sklearn import neighbors, datasets
#from matplotlib.colors import ListedColormap

## Create color maps for 3-class classification problem, as with iris
#cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features. We could
#                    # avoid this ugly slicing by using a two-dim dataset
#y = iris.target

#knn = neighbors.KNeighborsClassifier(n_neighbors=1)
#knn.fit(X, y)

#x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
#y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
#xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
#                        np.linspace(y_min, y_max, 100))
#Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

#Z = Z.reshape(xx.shape)
#plt.figure()
#plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

## Plot also the training points
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#plt.xlabel('sepal length (cm)')
#plt.ylabel('sepal width (cm)')
#plt.axis('tight')


import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                    # avoid this ugly slicing by using a two-dim dataset
y = iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])






Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('x - label')
plt.ylabel('y - label')
plt.title('Iris - data')

plt.show()

X = X1  # we only take the first two features. We could
                    # avoid this ugly slicing by using a two-dim dataset
y = y1

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])



Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('x - label')
plt.ylabel('y - label')
plt.title('random data')

plt.show()




