

import numpy as np 
import pandas as pn
from scipy import stats
from scipy.linalg import eig
import matplotlib.pyplot as plt
import numpy.random as nprnd
import select
import scipy.linalg as la

import math
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Zadanie 1
# wygenerować w sposób losowy zbiór 200 obiektów dwuwymiarowych za pomocą funkcji z numpy dot i rand lub randn

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)



rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
print("\n X = \n", X)

# zwizualizować obiekty na pomocą funkcji matplotlib, np. scatter : 

plt.scatter(X[:, 0], X[:, 1])
plt.title('random 2D data', fontsize = 15)
plt.show()

# dokonać redukcji do jednego wymiaru za pomocą własnej funkcji wiPCA,
# zwizualizować wektory własne oraz rzut wygenerowanych obiektów na pierwszą składową 

pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
#X_new = pca.inverse_transform(X_pca1)

t = pca.components_
a = pca.explained_variance_

pca = PCA(n_components=1)
pca.fit(X)
X_pca1 = pca.transform(X)

print("\nX_pca1\n", X_pca1)
#pca.components_[0,:] = -pca.components_[0,:]

print("\npca.components_\n", pca.components_)
print("\npca.explained_variance_\n", pca.explained_variance_)


print("\n t = \n", t)
print("\nX_pca\n", X_pca)
#pca.components_[0,:] = -pca.components_[0,:]
t[0,:] = -t[0,:]
print("\npca.components_\n", pca.components_)
print("\npca.explained_variance_\n", pca.explained_variance_)

X_new = pca.inverse_transform(X_pca1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.9, c='#ffc061')
for length, vector in zip(a, t):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.9, c='#ff9900')
plt.title('PCA, n_components = 2', fontsize = 15)
plt.axis('equal')
plt.show()

#ff9900


def wiPCA(data, n, q):

    #print("\n iris data =\n",data)   

    u = np.mean(data.T, axis=1) #średnia dla wierszy macierzy a   
   
    #print("\n u =\n", u)  # u = wektor średnich dla kolumn

    # normalizacja, standaryzacja zmiennych - (Macierz odchyleń)
    # od każdej wartości macierzy danych wejściowych odejmuje sumę wektora średnich (wektora u)
    data_norm = data - u

    #for i in range(data.shape[0]):  
    #    for j in range(data.shape[1]):           
    #        data_norm[i,j] = (data[i,j] - u[i])

    #print("\ndata_norm=\n", data_norm)  

    cov_matrix = np.cov(data_norm.T)

    print("\n cov_matrix =\n",cov_matrix)

    eigvals, eigvecs = la.eig(cov_matrix)

    eigvecs_sort = eigvecs

    #print("\nwartości własne = \n", eigvals)           
     
    if(n==1 and q==1):
        print("\n my if eles , n=1\n")
        eigvecs[:,0] = -eigvecs[:,0]        
        
        t2 = np.zeros(shape=(1,2))
        t2[0,:] = eigvecs[0,:]
        
        t1 = np.zeros(shape=(1,1))
        t1[0] = eigvals[0].real
        
        P = np.dot(data_norm, t2.T)        
        return P, t1, t2

    if(n==2 and q==1):
        
        eigvecs[:,0] = -eigvecs[:,0]
        P = np.dot(data_norm, eigvecs)        
        return P, eigvals, eigvecs

    if(n==2 and q==2):

        eigvecs[:,1] = -eigvecs[:,1]
        eigvecs[:,3] = -eigvecs[:,3]
      
    print("\nwektory własne = \n", eigvecs)

    print("\nsortownie wartości własnych =\n")
    w1 = -np.sort(-eigvals)
    w2 = np.argsort(-eigvals)    
    print(w1,"\n")    
   
    #for i in range(w2.size):
    #    print(eigvecs[w2[i],:])     
        
    print("")

    t = np.zeros(shape=(4,4))
    t1 = np.zeros(shape=(2,4))
    y = 4    

    #print("\n t = \n", t)

    #eigvecs[w2[0],:] = -eigvecs[w2[1],:]    
    #eigvecs[w2[3],:] = -eigvecs[w2[3],:] 
    
    print("wyznaczanie macierzy wektorów własnych na podstawie posortowanych wartości własnych =")
    #for i in range(2):    
    
    for i in range(y):
        t[i] = eigvecs[w2[i],:]
        #eigvecs_sort[i] = eigvecs[w2[i],:]

    #print("\n t = \n", t)
    
    
    #print("\n wektory własne   =\n", t)
    #print("\n t 0  =\n",t[0])
    #print("\n wektory własne - Transpozycja  =\n",t.T)
    
    print("")
    
    for i in range(2):
        t1[i] = t.T[i]        

    print("\n wybór wektorów własnych = \n", t1) 
    
    P = np.dot(data_norm, t1.T) # 2 wymiary
    
    #print("\n P = \n", P)

    return P, eigvals, eigvecs
#iris = datasets.load_iris()
#P = wiPCA(iris["data"],2)

targets = ['random 2D data', 'data reduced to one dimension']

P1, eigvals_1, eigvecs_1 = wiPCA(X,2,1)

print("P, n components = 1 \n", P1)
print("\n eigvals_p \n", eigvals_1)
print("\n eigvecs_p \n", eigvecs_1)

t = eigvecs_1
t[0,:] = -t[0,:]
a = eigvals_1

P2, eigvals_2, eigvecs_2 = wiPCA(X,1,1)

print("P, n components = 1 \n", P2)
print("\n eigvals_p \n", eigvals_2)
print("\n eigvecs_p \n", eigvecs_2)

#t = pca.components_
#a = pca.explained_variance_


#print("\n pca.explained_variance_ \n", pca.explained_variance_)
#print("\n pca.components_ \n", pca.components_)

#X_new = pca.inverse_transform(X_pca1)


#X_new = pca.inverse_transform(P)

#X_new = pca.inverse_transform(X_pca)
X_new = pca.inverse_transform(P2)

plt.scatter(X[:, 0], X[:, 1], alpha=0.9, c='#7a9cff')

for length, vector in zip(a, t):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.9, c='#0040ff')
plt.title('wiPCA, n = 2', fontsize = 15)
plt.legend(targets)
plt.axis('equal')
plt.show()

# zadanie 2 :
iris = datasets.load_iris()

P, a, b = wiPCA(iris["data"],2,2)
print("\nP, n components = 2\n", P)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pn.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
print("\nP df =\n", df)

principalDf = pn.DataFrame(data = P, columns = ['principal component 1', 'principal component 2'])

print("\nP data frame =\n", principalDf)

finalDf = pn.concat([principalDf, df[['target']]], axis = 1)

print("\nP finalDf =\n", finalDf)

# Visualize 2D Projection
print("\nP Visualize 2D Projection =\n")

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title('wiPCA, n_components = 2', fontsize = 15)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()