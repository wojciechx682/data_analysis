import numpy as np 
import pandas as pn
from scipy import stats
import matplotlib.pyplot as plt
from io import StringIO
import random 
from numpy.random import randint

a = pn.DataFrame(np.random.normal(size=(5,3)), index = pn.date_range(start='3/1/2020', end='3/5/2020'), columns=['A','B','C'])
a.index.name="data"
print(a, "\n")

b = pn.DataFrame(np.random.randint(1, 5, size=(20, 3)), index = np.arange(1,21,1), columns=['A','B','C'])
b.index.name="id"
print(b, "\n")

print(b.iloc[0:3], "\n") 
print(b.head(3), "\n") 

print(b.iloc[-3:], "\n")
print(b.tail(3), "\n") 

print(b.index.name, "\n")  

print(b.columns.names, "\n")
print(b.columns, "\n")
 
print(b.values, "\n")  
print(b.sample(n=5), "\n")  

print(b.values[:,0], "\n")  
print(b.values[:,0:2], "\n")

print(b.iloc[0:3], "\n") 
print(b.iloc[:,0:2], "\n") 

print(b.iloc[4], "\n") 

print(b.iloc[[0,4,5,6],[0,1]], "\n") 

#  Zapoznaj się z funkcją ’describe’ i wyświetl podstawowe statystyki tabeli:

print(b[b>0], "\n") # sprawdź które dane są są większe od 0

#wyświetl tylko dane większe od 0,
print(b.values[b>0], "\n")

# wybierz z kolumny A wartości > 0,
print(b.A[b.A>0], "\n")

# policz średnią w kolumnach
print(b.mean(axis=1), "\n")       # 'Parameters: axis{index (0), columns (1)}'

# policz średnią w wierszach
print(b.mean(axis=0), "\n")

a = pn.Series([11,12,13],[14,15,16])
b = pn.Series([17,18,19])
c = pn.concat([a, b], keys=['a', 'b'])
print(a, "\n")
print(b, "\n")
print(c, "\n")

print("Dokonaj transpozycji nowej tabeli \n") 
print(c.transpose(), "\n")

# Sortowanie
# – posortuj dane po ’id’ rosnąco,
# – posortuj dane po kolumnie ’y’ malejąco.

df = pn.DataFrame(([1, 2, 3, 4, 5], ['a', 'b', 'a', 'b', 'b']), index=np.arange(0,2,1))
#df = pn.DataFrame((x=[1, 2, 3, 4, 5], y=["a", "b", "a", "b", "b"]), index=np.arange(5))
#df = pn.DataFrame("x": [1,2,3,4,5], "y": ['a', 'b', 'a', 'b', 'b'])
df.index.name='id' 
print(df)

print("\n-------------------\n")

b = [1, 2, 3, 4, 5]
b = np.resize(b,(5,1)) 
print(b)

print(" ")

c = ['a', 'b', 'a', 'b', 'b']
c = np.resize(c,(5,1))         
print(c)


df = pn.DataFrame(b)
print(" ")
print(df)

####################
print(" ")

#my_array = ([[1, 2, 3, 4, 5]],[['a', 'b', 'a', 'b', 'b']])
a = np.array([[1,2,3,4,5], ['a', 'b', 'a', 'b', 'b']])
a = np.transpose(a)
print(a)

# Nieco zmodyfikowałem polecenie :
# df = pd.DataFrame(”x”: [1, 2, 3, 4, 5], ”y”: [’a’, ’b’, ’a’, ’b’, ’b’], index=np.arange(5))
# ponieważ nie chciało mi ono działać, zamiast tego stworzyłem zamierzony obiekt data frame w inny sposób :

df = pn.DataFrame(a, index=randint(1,10,5), columns=['X','Y'])
df.index.name='id' 
print(" ")
print(df)

# Sortowanie danych według 'id' - rosnąco 
print(" ")
print(df.sort_index())

# posortuj dane po kolumnie ’y’ - malejąco :
print(" ")
print(df.sort_values(by=['Y'], ascending=False))

print("\n-------------------\n")

# Grupowanie danych 

#slownik = 'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'], 'Fruit': ['Apple', 'Apple', 'Banana', 'Banana', 'Apple'], 'Pound': [10, 15, 50, 40, 5], 'Profit':[20, 30, 25, 20, 10]

słownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'], 'Fruit': ['Apple', 'Apple', 'Banana', 'Banana', 'Apple'],'Pound':[10,15,50,40,5],'Profit':[20,30,25,20,10]}
df3 = pn.DataFrame(słownik) # utworzy obiekt data frame ze zmiennej słownik
print(df3)
print("\n", df3.groupby('Day').sum()) # wypisze sumę wartości 'Pound' i 'Profit' dla poszczególnych dni ( dla wszystkich owoców)
print("\n", df3.groupby(['Day','Fruit']).sum()) # wypisze sumę wartości 'Pound', 'Profit' dla poszczególnych owoców, dla poszczególnych dni

# Wypełnianie danych

df=pn.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
df.index.name='id'
print("\n", df)

# Wykonaj i opisz jak działają poniższe komendy:

df['B']=1       # zamienia wszystkie wartości kolumny 'B' na '1'
print("\n", df)

df.iloc[1,2]=10 # Wstawia wartość '10' do trzeciej kolumny drugiego wiersza (indeks wiersza = 1. indeks kolumny = 2)
print("\n", df)

df[df!=0]=-df   # Zamienia wszystkie liczby na przeciwne
print("\n", df)

# Uzupełnianie danych

df.iloc[[0,3],1] = np.nan # Wpisuje wartość 'NaN' (Not a Number) do pierwszego i czwartego wiersza, z kolumny drugiej
print("\n", df)

df.fillna(0, inplace=True) # Zastąpi wartości 'NaN' zerami
print("\n", df)

df.iloc[[0,3],1] = np.nan
df = df.replace(to_replace=np.nan,value=-9999) # Zastąpi wartości 'NaN' wartością '-9999'
print("\n", df)

df.iloc[[0, 3], 1] = np.nan 
print("\n", pn.isnull(df), "\n") # Wypisze 'True' dla wartości 'NaN' oraz 'False' dla pozostałych wartości

print("\n---------------------------------")
print("\nZADANIE 1\n")

# Zgrupować tabele po zmiennej symbolicznej X,
# a następnie 
# wyznaczyć średnią wartość atrybutu numerycznego Y w grupach wyznaczonych przez X,

data = np.array([[1,2,3,4,5], ['a', 'b', 'a', 'b', 'b']])
data = np.transpose(data)
print("\n", data)

df = pn.DataFrame({'X': [1,2,3,4,5], 'Y': ['a', 'b', 'a', 'b', 'b']})
print(" ")
print(df)

print(df.groupby(['Y']).mean())
#print(df.groupby(['Y']))

print("\n---------------------------------")
print("\nZADANIE 2\n")

print(df.value_counts(), "\n")

print(df.value_counts('X'), "\n\n") # Liczebność wartości atrybutu X

print(df.value_counts('Y'), "\n\n") # Liczebność wartości atrybutu Y


print("\n---------------------------------")
print("\nZADANIE 3\n")

data1 = np.loadtxt('autos.csv', dtype='str') 
print(" ")
print(data1)

data = pn.read_csv('autos.csv') # funkcja read_csv automatycznie dopasowuje zawartość do formatu data frame
print(" ")
print(data)

df = pn.DataFrame(data)
print(" ")
print(df)

print("\n---------------------------------")
print("\nZADANIE 4\n")

df = df.groupby(['make']).mean()
sum_column = df["city-mpg"] + df["highway-mpg"]
df["srednie_zuzycie"] = (sum_column/2)
print(df.iloc[:,-1], "\n")      # Wyświetli średnie wartości zużycia paliwa dla każdego z prodecuntów
print(df.iloc[:,[13,14]], "\n") # Wyświetli średnie zużycie paliwa dla każdego z prodecuntów (odpowiednio: średnie zużycie w mieście, i poza miastem)

print("\n---------------------------------")
print("\nZADANIE 5\n")

data = pn.read_csv('autos.csv')
df = pn.DataFrame(data)
df = df.groupby(['make','fuel-type']).size()
print(df)  # Wyświetli liczności atrybutu fuel-type dla każdego modelu

print(" ")

data = pn.read_csv('autos.csv')
df = pn.DataFrame(data)
print(df.value_counts('fuel-type')) # Wyświetli liczności atrybutu fuel-type
print(" ")


# inny sposób : 
data = pn.read_csv('autos.csv')
df = pn.DataFrame(data)
df = df.iloc[:,[2,3]] # przechouje zawartość kolumn 'make' i 'fuel-type' z data frame
print(df.value_counts(ascending=True), "\n") # Wyświetli liczności atrybutu 'fuel-type' dla poszczególnych modeli
print(df.value_counts('fuel-type', ascending=True)) # Wyświetli liczności atrybutu fuel-type

print("\n---------------------------------")
print("\nZADANIE 6, 8\n")

data = pn.read_csv('autos.csv')
df = pn.DataFrame(data)

length = df.values[:,10]    
city_mpg = df.values[:,23]  

length = length.astype(float)
city_mpg = city_mpg.astype(float)

poly1 = np.polyfit(length,city_mpg,1) 
poly2 = np.polyfit(length,city_mpg,2) 
#poly3 = np.polyfit(length,city_mpg,3) 

print("\n", poly1) # Wyświetli współczynniki wielomianu (jest ich tyle ile wynosi stopień wielomianu...)
print("\n", poly2)

# aby wyświetlić wartość wielomianu dla konkretnego argumentu (x) należało by skorzystać z polyval ...


p1 = np.poly1d(poly1)
p2 = np.poly1d(poly2)

xp = np.linspace(140,208,205)

#plt.plot(length,city_mpg,'.',xp,p1(xp),'-')  
plt.plot(length,city_mpg,'.',xp,p2(xp),'g-')  
plt.legend(['city_mpg(length)','wielomian aproksymujacy'])
plt.title('ZADANIE 6, 8')
plt.show()

print("\n---------------------------------")
print("\nZADANIE 7\n")

print(stats.pearsonr(length, city_mpg))
print(stats.pearsonr(length, city_mpg)[0]) # zwróci współczynnik korelacji Pearsona

print("\n---------------------------------")
print("\nZADANIE 9\n")

data = pn.read_csv('autos.csv')
df = pn.DataFrame(data)

length = df.values[:,10] 
length = length.astype(float)

density = stats.gaussian_kde(length)
x_axis_density = np.linspace(100,250,205) 
x_axis_length =  np.zeros(205, dtype='int')

# Wizualizacja próbek i funkcji gęstości na dwóch wykresach :
#fig, axs = plt.subplots(2)   
#axs[0].plot(x_axis_density,density(x_axis_density), '-')
#axs[0].set_title('funkcja gęstości')
#axs[1].plot(x_axis_length, length, '.')
#axs[1].set_title('próbki ')

plt.plot(length, x_axis_length, '.', x_axis_density,density(x_axis_density), 'g-',)
plt.legend(['length','funkcja gęstości dla zmiennej length'])
plt.title('ZADANIE 9')
plt.show()


print("\n---------------------------------")
print("\nZADANIE 10\n")

data = pn.read_csv('autos.csv')
df = pn.DataFrame(data)

length = df.values[:,10] 
length = length.astype(float)

width = np.array(df.values[:,11]) 
width = width.astype(float)

print(np.sort(width))

density_length = stats.gaussian_kde(length)
density_width = stats.gaussian_kde(width)
x_axis_density = np.linspace(100,250,205) 
x_axis_density1 = np.linspace(60,75,205) 
x_axis_length =  np.zeros(205, dtype='int')
x_axis_width =  np.zeros(205, dtype='int')

# Wizualizacja próbek i funkcji gęstości na dwóch wykresach :
fig, axs = plt.subplots(2)   
axs[0].plot(length, x_axis_length, '.', x_axis_density, density_length(x_axis_density), 'g-')
axs[0].set_title('funkcja gęstości zmiennej length')
axs[1].plot(width, x_axis_width, '.', x_axis_density1, density_width(x_axis_density1), 'g-')
axs[1].set_title('funkcja gęstości zmiennej width')

#plt.plot(length, x_axis_length, '.', x_axis_density,density(x_axis_density), 'g-')
#plt.legend(['length','funkcja gęstości dla zmiennej length'])
#plt.title('ZADANIE 9')
plt.show()


print("\n---------------------------------")
print("\nZADANIE 11 -\n")

data = pn.read_csv('autos.csv')
df = pn.DataFrame(data)

x = np.array(df.values[:,10]) 
x = x.astype(float)

y = np.array(df.values[:,11]) 
y = y.astype(float)

xy = np.vstack((x, y))

density = stats.gaussian_kde(xy)

x_min = x.min()
x_max = x.max()

y_min = y.min()
y_max = y.max()

size = 205

x_flat_width = np.linspace(x_min, x_max, size)
y_flat_length = np.linspace(y_min, y_max, size)

x, y = np.meshgrid(x_flat_width, y_flat_length)

xy = np.dstack((x, y))

z = np.apply_along_axis(density, 2, xy)

z = z.reshape(205,205)

plt.plot(z)
plt.title('dwuwymioarowy estymator funkcji gęstości, dla zmiennych width i length')
plt.savefig('zadanie_11.png')
plt.savefig('zadanie_11.pdf')
plt.show()
