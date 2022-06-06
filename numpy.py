 
import numpy as np
import random 
 
from numpy.random import seed
from numpy.random import rand
from numpy.random import randint
from numpy.lib.stride_tricks import as_strided
import array as arr

seed(1)

# arange # linspace # random # zeros # ones # shape # reshape # sort # argsort # dot # strides

# Tablica jenowymiarowa
a = np.array([1,2,3,4,5,6,7])

# Tablica dwuwymiarowa
b = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print(a)
print("\n", b)

# Wykonaj transpozycję tablicy ’b’ za pomocą funkcji transpose.
b = np.transpose(b)
print("\ntranspozycja: \n", b, "\n")

# Utwórz i wyświetl tablicę składającą się ze 100 elementów za pomocą funkcji arange.

c = np.arange(100)
print(c)

# Utwórz i wyświetl tablicę składającą się z 10 liczb w zakresie od 0 do 10

d = np.arange(0,10,1)
print("\n d = \n ")
print(d)

# Użyj funkcji linspace

e = np.linspace(1, 3, num=3, dtype=int)
print("\n e = \n", e)
 
# Za pomocą arange utwórz tablicę pomiędzy wartościami od 0 do 100 i skoku wartości co 5

f = np.arange(0,100,5)
print("\n f = \n ")
print(f)

# LICZBY LOSOWE

# Za pomocą funkcji random utwórz tablicę z 20 liczb losowych rozkładu normalnego, zaokrąglonych do dwu miejsc po przecinku.

g = np.random.randn(4,5)
np.round(g, 2, g)
print("\n g = \n", g)

# Wygeneruj losowo 100 liczb całkowitych w zakresie od 1 do 1000

rand_numbers = random.sample(range(1, 1000), 100)
print("\nrand_numbers = \n", rand_numbers)

# Wygeneruj losowo 100 liczb całkowitych w zakresie od 1 do 1000 (inny sposób) :

rand_values = randint(1,1000,100,int)
print("\nrand values = \n", rand_values)

#rand_values1 = randint(1,5,(5,5))

# Utwórz macierz losową złożoną z liczb całkowitych o rozmiarze 5x5 i nadaj jej typ 32bit
a = np.random.randint(5, size=(5, 5), dtype=np.int32)
print("\n", a)

# Za pomocą funkcji ’zeros’ i ’ones’ wygeneruj dwie macierze o rozmiarze 3x2.

# matrix_1 = np.zeros()
#a = np.array([ 1.,  2.,  3.], dtype=np.float32)
#print(a)

matrix_1 = np.zeros((3,2), dtype=int)
matrix_2 = np.ones((3,2), dtype=int)

print("\nzeros =\n", matrix_1, "\n")
print("ones =\n", matrix_2)

# Wygeneruj tablicę złożoną z losowo wybranych liczb dziesiętnych od 0-10 (a).
    
a = np.array((np.random.uniform(0,10,10)))
print("\na = \n", a)

a = a.astype(np.int32)
print("\n a (int) = \n", a)

#np.append(a, [1, 2, 3, 4])

print("\n złączone tablice =\n")
print(np.append(a, [1, 2, 3, 4]))

#q = np.random.uniform(0,10,10)
#print("\nq = \n", q)

#z = np.arange(0,10,0.1)
#print("\nz = \n", z)

###########################################################

#SELEKCJA DANYCH:
print("\nSELEKCJA DANCYH : ")
b = np.array([[1,2,3,4,5], [6,7,8,9,10]], dtype=np.int32)
print(" ")
print("b = \n",b)

# DANE ZWIĄZANE Z MACIERZĄ "b" 
print("\nWymiar tablicy b = ",b.ndim)
print("Rozmiar tablicy b = ",b.size)
print("Elementy tablicy o wartosci 2 i 4 :",b[0,1],b[0,3])
print("Pierwszy wiersz tablicy : ",b[0,:])
print("Wszystkie wiersze z kolumny pierwszej :",b[:,0])

# Wygeneruj macierz losową o rozmiarze 20x7, złożoną liczb całkowitych w przedziale 0-100.
M = randint(0,100,(20,7))
print("\n", M)
print("\nWszystkie wiersze dla 4 pierwszych kolumn :\n",M[:,0:4])

###########################################################

# OPERACJE MATEMATYCZNE I LOGICZNE
# Stwórz dwie macierze w przedziale 0-10 o rozmiarach 3x3 (a i b).
# Dodaj, pomnóż, podziel, spotęguj je przez siebie 

seed(1) 

a = randint(0,10,(3,3))
print("\na= \n", a)

b = randint(0,10,(3,3))
print("\nb= \n", b)

print("\nSUMA MACIERZY : \n", np.add(a,b))
print("\nILOCZYN MACIERZY : \n", np.matmul(a,b))                 
print("\nILORAZ MACIERZY : \n", np.divide(a,b)) 
print("\nPOTEGA MACIERZY : \n", np.power(a,b))
print("\nWARTOŚCI >= 4 : \n", a[a>=4])
print("\nWARTOŚCI >= 4 : \n", np.greater_equal(a, 4))
print("\nWARTOŚCI >= 1, <= 4 : \n", np.logical_and(np.greater_equal(a, 1), np.less_equal(a, 4)))
print("\nSUMA GŁÓWNEJ PRZEKĄTNEJ MACIERZY B : ", np.trace(b))

###########################################################

# DANE STATYSTYCZNE:
print("\nSUMA ELEMENTÓW MACIERZY B : ", np.sum(b))
print("\nMINIMALNA WARTOŚĆ MACIERZY B : ", np.amin(b))
print("\nMAKSYMALNA WARTOŚĆ MACIERZY B : ", np.amax(b))
print("\nODCHYLENIE STANDARDOWE MACIERZY B : ", np.std(b))
print("\nŚREDNIA DLA WIERSZY W MACIERZY B : ", b.mean(1))
print("\nŚREDNIA DLA KOLUMN W MACIERZY B : ", b.mean(0))

###########################################################

# RZUTOWANIE WYMIARÓW ZA POMOCĄ SHAPE LUB RESIZE:

# Utwórz tablicę składającą się z 50 liczb:
a = np.arange(50)
a = a.reshape(10,5)
print("\na= \n", a)

b = np.arange(50)
b = np.resize(b,(10,5)) 

print("\nb= \n", b)

# ravel - funkcja "splaszczajaca" macierz (zwrócenie jedno-wymiarowej tablicy)

x = np.array([[1, 2, 3], [4, 5, 6]])
a = np.ravel(x)

print("\n",a)

# Stwórz dwie tablice o rozmiarach 5 i 4 i dodaj je do siebie. Sprawdź do czego służy funkcja ’NEWAXIS’ i wykorzystaj ją

print("-----------------------------------------------")
print("Stwórz dwie tablice o rozmiarach 5 i 4 i dodaj je do siebie.")

a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9])

#np.add(a,b)
print("a + b = ", np.append(a,b))

#print(array3)

print("-----------------------------------------------")

print("Sprawdź do czego służy funkcja ’NEWAXIS’ i wykorzystaj ją")

# funkcja newaxis służy do zmiany wymiarów tablicy (1D -> 2D, 2D -> 3D ... )

a = np.array([1,0,8,3])
a=a[:, np.newaxis]
print("\na = \n", a)

print("-----------------------------------------------")
print("SORTOWANIE DANYCH ")

###########################################################
# SORTOWANIE DANYCH 

print(np.sort([4,6,1,9]))
print(np.argsort([4,6,1,9])) 

#print(np.array([[1,3,6,2],[8,9,5,4]]))

a = np.random.randn(5,5)

# posortuj wiersze rosnąco. 

a=np.sort(a,0)
print("\n", a)

# posortuj kolumny malejąco.

b=([[1,3,5,7], [2,2,6,0]])
print("\n", sorted(b, reverse=True))

# Na podstawie poniższej tablicy zrób macierz 3x3:

b = np.array([(3,'MZ','mazowieckie'),
            (2,'ZP','zachodniopomorskie'),
            (1,'ML','małopolskie')])

A = [ [b[0,0],b[0,1],b[0,2]],
      [b[1,0],b[1,1],b[1,2]],
      [b[2,0],b[2,1],b[2,2]],]

A = np.resize(A,(3,3))
#A.reshape(3,3)

# lub (inny sposób) :
B = np.matrix(b)

print("\nA = \n", A)
print("\nB = \n", B)

# Posortuj dane rosnąco po kolumnie 2:

print("\n", A[A[:,1].sort()])
print("\n", A[1,2])

print("-----------------------------------------------")

# ZADANIA PODSUMOWUJĄCE

print("\nZADANIE 1")

matrix = randint(0,10,(10,5))
print("\n", matrix)
 
print("\nWARTOŚCI NA GŁÓWNEJ PRZEKĄTNEJ : ", np.diag(matrix))
#print("\nWARTOŚCI NA GŁÓWNEJ PRZEKĄTNEJ : ", np.diagonal(matrix))
print("SUMA GŁÓWNEJ PRZEKĄTNEJ : ", matrix.trace())

print("-----------------------------------------------")

print("\nZADANIE 2\n")

a = np.random.normal(0, 0.1, 5)
b = np.random.normal(0, 0.1, 5)

print(a, "\n\n", b, "\n")

print(np.multiply(a,b)) 

print("-----------------------------------------------")

print("\nZADANIE 3")

a = random.sample(range(1, 100), 25)
b = random.sample(range(1, 100), 25)

#res = np.append(a, b)

print("\n",a)
print("\n",b)
#print("\n",res)

a1 = np.resize(a,(5,5))
b1 = np.resize(b,(5,5))

print("\n",a1)
print("\n",b1)
print(" ")

print(np.add(a1,b1))

print("-----------------------------------------------")

print("\nZADANIE 4")

#matrix_2 = np.randn((4,5), dtype=int)
#print(matrix_2)

a = randint(1,10,(4,5))
print("\na = \n", a)

b = randint(1,10,(5,4))
print("\nb = \n",b)

b = b.reshape(4,5)
print("\nb (reshaped) = \n", b)

print("\nSUMA MACIERZY :\n",np.add(a,b))

print("-----------------------------------------------")

print("\nZADANIE 5")

print("\nKolumna 3 macierzy a :",a[:,2]) # to można wykomentować
print("\nKolumna 4 macierzy b :",b[:,3])
print("\nIloczyn kolumn: ",a[:,2]*b[:,3])
print("\nILOCZYN KOLUMN: 3 (macierzy a) i 4 (macierzy b) = ", np.matmul(a[:,2],b[:,3]))

print("-----------------------------------------------")

print("\nZADANIE 6\n")

a = np.random.normal(0, 1, (3,3))
b = np.random.uniform(0, 1, (3,3))

print(a)
print("\nŚrednia = ", a.mean())
print("Odchylenie standardowe = ", a.std())
print("Wariancja = ", a.var())

print("\n", b)
print("\nŚrednia = ", b.mean())
print("Odchylenie standardowe = ", b.std())
print("Wariancja = ", b.var())

print("-----------------------------------------------")

print("\nZADANIE 7\n")

a = randint(1,10,(2,2))
b = randint(1,10,(2,2))

print("\na = \n", a)
print("\nb = \n", b)

print("\nIloczyn = \n",a*b)
print("\nIloczyn (dot) = \n",np.dot(a,b))

# Funkcja dot wygenerowała poprawny wynik będący iloczynem macierzy a i b, 
# natomiast mnożenie w postaci a*b nie dało poprawnego wyniku (to nie jest mnożenie macierzy, tylko odpowiadającym sobie elementom z tych macierzy (a[0,0] * b[0,0] itd...))

print("-----------------------------------------------")

print("\nZADANIE 9\n")
a = np.array([6.31, 0.67, 1.77, 9.55])
b = np.array([3.48, 1.90, 3.61, 7.27])

print(np.vstack((a,b)))
print("\n", np.hstack((a,b)))

# Czym one się różnią? Zastanów się i napisz, w jakich przypadkach warto je zastosować?

# funkcja vstack - łaczy tablice w kolejności wertykalnej (pionowo),
# natomiast hstack - horyzontalnie (poziomo)

print("-----------------------------------------------")

print("\nZADANIE 8, 10\n")

# Sprawdź funkcję strides oraz as strided. Wykorzystaj je do wyboru danych z macierzy np. 5 kolumn z trzech pierwszych wierszy.
# Użyj funkcji strides i as strided do obliczenia wartości maksymalnej bloków danych z macierzy (rysunek)

array = np.array([[0,1,2,3,4,5],
                  [6,7,8,9,10,11],
                  [12,13,14,15,16,17],
                  [18,19,20,21,22,23]])

print("\narray= \n", array)
print("\narray.strides -> ", array.strides)

a0 = as_strided(array, strides=(24,4), shape=(3,5))          # <--- 5 kolumn z 3 pierwszych wierszy
a1 = as_strided(array[0:2,0:3], strides=(24,4), shape=(2,3)) # BLOK 1
a2 = as_strided(array[0:2,3:6], strides=(24,4), shape=(2,3)) # BLOK 2
a3 = as_strided(array[2:4,0:3], strides=(24,4), shape=(2,3)) # BLOK 3
a4 = as_strided(array[2:4,3:6], strides=(24,4), shape=(2,3)) # BLOK 4  

print("\nBLOK 3x5 =\n", a0)     
print("\nBLOK 1 =\n", a1)     
print("\nBLOK 2=\n", a2)    
print("\nBLOK 3 =\n", a3)      
print("\nBLOK 4 =\n", a4)      
print("\nWARTOŚĆ MAKSYMALNA Z BLOKU 1 = ", a1.max())     
print("\nWARTOŚĆ MAKSYMALNA Z BLOKU 2 = ", a2.max())    
print("\nWARTOŚĆ MAKSYMALNA Z BLOKU 3 = ", a3.max())    
print("\nWARTOŚĆ MAKSYMALNA Z BLOKU 3 = ", a4.max())   