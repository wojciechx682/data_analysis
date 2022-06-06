

import numpy as np 
import pandas as pd
from scipy import stats
from scipy.linalg import eig
import matplotlib.pyplot as plt
import numpy.random as nprnd
import select
import scipy.linalg as la
import random


import math
from math import*
from decimal import Decimal
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

def dist(X, C):
    return np.sqrt(np.add.outer(np.sum(X*X, axis=1), np.sum(C*C, axis=1)) - 2 * X.dot(C.T))

def ksrodki(X, k_number, a):

    C = np.array(np.random.randint((k_number*2), size=(k_number,2)))
    CN = np.zeros((k_number,2))



    for k in range(15):
    #print("\n X = \n", X)
    #print("\n C = \n", C)        

        d = dist(X, C)
        print("\n dist = \n", d)

        CN = C
        print("\n CN = \n", CN)

        # Sąsiedztwem środka cj nazywamy zbiór wszystkich punktó xi,
        # któe należą do jego sąsiedztwa

        s = np.argmin(d, axis=1) # sąsiedztwo
        print("\n s = \n", s)

        s_argsort = np.argsort(s)
        print("\n s - argsort = \n", s_argsort)

        for i in range(k_number):
            result = np.where(s == i) # Ci = 0, 1, 2  (trzy punkty)
            #print("\n result = \n", result)
            # przynależności punktów Xi do centrów Ci : 
            #print("\n C[0] -> = ", result[0])
    
            count = 0
            for element in result[0]:
                count += 1
            print("\n number of elements in result = ", count)

            t = count
            ashape = (t,2)  

            M = np.zeros(ashape)
            #print("\n M = \n", M)

            for j in range(count):
                M[j] = X[result[0][j]]
                #M[j] = [3,5]

            print("\n M = \n", M)

            M_mean = np.mean(M, axis=0)
            print("\n  M_mean = \n", M_mean)

            print("\n result length = ", int(len(result[0])))

            print("\n C[", i, "] ->",result[0])

       
            if(len(result[0]) == 0):
                print("\m M_Mean is NaN \n")
                #CN[i] = np.array(np.random.randint(0, 55, size=(1,2)))
                CN[i] = np.mean(C, axis=0)
            else:   
                CN[i] = M_mean

            #isNaN = np.isnan(M_mean)
            #if(isNaN):
            #   print("M_mean is NAN",isNaN)         

            print("\n CN = \n", CN)

            plt.scatter(M[:,0],M[:,1])
    
            #plt.scatter(X[:,0],X[:,1],c='g')    

            #print("\n C[1] -> = ", result[0])
            #print("\n C[2] -> = ", result[0])
        #plt.scatter(CN[:,0],CN[:,1],c='r')
        
        colors = ["blue","orange","green","red","purple"]


        for j in range(k_number):
            plt.scatter(CN[j,0],CN[j,1], c=colors[j], s=225)
        #plt.scatter(CN[1,0],CN[1,1], c='orange', s=125)
        #plt.scatter(CN[2,0],CN[2,1], c='green', s=125)
        #plt.scatter(CN[3,0],CN[3,1], c='red', s=125)

        if(a==True):
            plt.title("autos.csv")
            plt.xlabel("city-mpg")
            plt.ylabel("highway-mpg")
            plt.show()
        else:
            plt.title("random_data")
            plt.xlabel("x-label")
            plt.ylabel("y-label")
            plt.show()

        C = CN

#X = np.array(np.random.randint(300, size=(150,2)))
#C = np.array(np.random.randint(6, size=(3,2)))

data = pd.read_csv('autos.csv') # funkcja read_csv automatycznie dopasowuje zawartość do formatu data frame
df = pd.DataFrame(data)
print("\n autos.csv = \n", df)

Y = np.array(np.random.randint(0, 150, size=(1,2)))

print("\n Y = ", Y)

a = df.iloc[:,23].values
b = df.iloc[:,24].values
print("\n a = \n", a)
print("\n b = \n", b)

X = df[["city-mpg", "highway-mpg"]].to_numpy()

print("\n X = \n", X)

print("\n użycie funkcji ksrodki -> \n")

ksrodki(X,2,True)

print("\n użycie funkcji ksrodki -> \n")

ksrodki(X,5,True)

#df = pd.DataFrame(data)


#for i in range(len(X)):
#    for k in range(len(C)):                      
#        #print(np.sqrt(np.sum(((X[i]-C[k])**2)*V1 )))
#         print(np.sqrt(np.sum((X[i]-C[k])*(V**(-1))*np.transpose(X[i]-C[k])))) 

print("-----------------------------------------------------------------------------------------")

#X = np.array([[2, 2], [1.5, 1.5], [4, 8], [4.5, 8.5], [5, 5], [4.5, 4.5]])
#X = np.array([[1.5, 1.5],   [5, 5], [4.5, 4.5], [2, 2], [4.5, 8.5], [4.9, 3.7], [4, 8], [2.25, 0.7], [4.5,7]])


#C = np.array([[1, 1], [3, 7], [6, 6]])
#C = np.array([[5.0, 5.0], [9.0, 9.0], [12.0, 12.0]])

#CN = np.zeros((4,2))

#print("\n X = \n", X)
#print("\n C = \n", C)
#print("\n CN = \n", CN)

#d = dist(X, C)
#print("\n dist = \n", d)

# Sąsiedztwem środka cj nazywamy zbiór wszystkich punktó xi,
# któe należą do jego sąsiedztwa

#s = np.argmin(d, axis=1) # sąsiedztwo
#print("\n s = \n", s)

#s_argsort = np.argsort(s)
#print("\n s - argsort = \n", s_argsort)

# zwróc Xi dla danego Ci :
# np. zwróc te punkty Xi, gdzie Ci = 0

#result = np.where(s == 0)
#print("\n result = \n", result)
#print("\n result = \n", result[0])


#M = np.array([X[0], X[1]])
#print("\n  M = \n", M)

#M_mean = np.mean(M, axis=0)
#print("\n  M_mean = \n", M_mean)

#CN[0] = M_mean

#print("\n CN[0] = \n", CN[0])

#print("\n CN = \n", CN)


#print("")
#i=0
#while(s[i] == 0):    
#    i=i+1

#M = np.zeros((i,2)) #macierz przechowująca punkty należące co centrum CN[i]
#print("\n  M = \n", M)
#print("")

#i=0
#while(s[i] == 0):
#    #M = np.zeros((i+1,2))
#    print(X[i])
#    M[i] = X[i]

#    i=i+1

#print("\n  M = \n", M)
#print("")

#M_mean = np.mean(M, axis=0)
#print("\n  M_mean = \n", M_mean)

#CN[0] = M_mean

#print("\n CN[0] = \n", CN[0])

#print("\n CN = \n", CN)

print("-----------------------------------------------------------------------------------------")
print("moja DUŻA PĘTLA:")

#print("\n X = \n", X)
#print("\n V = \n", V)
#print("\n C = \n", C)

#plt.scatter(X[:,0],X[:,1])
#plt.scatter(C[:,0],C[:,1],c='r')
#plt.show()

    #d = dist(X, C)
    #print("\n dist = \n", d)


# Sąsiedztwem środka cj nazywamy zbiór wszystkich punktó xi,
# któe należą do jego sąsiedztwa

X = np.array(np.random.randint(900, size=(300,2)))
print("\n X = \n", X)
ksrodki(X, 2, False)

print("\n X = \n", X)
ksrodki(X, 5, False)

#C = np.array(np.random.randint(6, size=(3,2)))
#CN = np.zeros((3,2))


#print("\n X = \n", X)
#print("\n V = \n", V)
#print("\n C = \n", C)




#plt.scatter(X[:,0],X[:,1])
#plt.scatter(C[:,0],C[:,1],c='r')
#plt.show()




#print("\n C = \n", C)










#s = np.argmin(d, axis=1) # sąsiedztwo
#print("\n s = \n", s)
#print("\n len C = \n", len(C))

#print("-----------------------------------------------------------------------------------------")

#print("\n X = \n", X)

#print("\n C = \n", C)
#print("\n CN = \n", CN)

#print("\n result[0][0] = \n", result[0][0])
#print("\n result[0][1] = \n", result[0][1])

#print("\n X[1] = \n", X[result[0][0]])

#print("\n X[0] = \n", X[0])


#print("\n M = \n", M)
