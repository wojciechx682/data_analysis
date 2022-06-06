import numpy as np 
import pandas as pn
from scipy import stats
import matplotlib.pyplot as plt
import math

# ENTROPIA - miara rozproszenia informacji 

def freq(x, prob):
    # x - nazwa kolumny (string)
    xi = np.unique(data[x])

    if(prob==True):
        pi = (data.value_counts(x).values)/data[x].count()
        return xi, pi
    else:        
        ni = data.value_counts(x).values
        return xi, ni

def freq2(x, y, prob):

    a = np.array([data[x].values, data[y].values]) # tworzę tablicę z wartościami wybranych kolumn
    a = np.transpose(a) # transpozycja - zamiana rozmiaru

    #print("\na = \n")
    #print(a)
    
    df = pn.DataFrame(a, columns=[x,y]) # tworzę obiekt data frame z tej tablicy
    
    #print("\ndf = \n")
    #print(df)

    #print("\ndf-unique = \n")
    #print(df["type"].unique())

    #df = df.drop_duplicates()
    #print("\ndf = \n")
    #print(df.values)

    #print("\ndf dtypes= \n")
    #print(df.dtypes)
    #print("\ndrop duplicates = \n")
    #print("")
    #print(df.drop_duplicates().values[:,[0,1])
    #print("")
        #xi = df.drop_duplicates().values[:,[0,1]]
    #print("")
    #xi = np.unique(a, axis=1) # unikalne wartości w macierzy (wiersze)    # TYLKO TO ZMIENIĆ ! PI JEST DOBRZE WYZNACZONE
    #print("\nxi = \n")
    #print(xi)

    if(prob==True):            
        #pi = (df.value_counts().values)/(df[x].size)          
        pi = df.groupby([x,y]).size().reset_index(name='Count')         
  
        xi = pi.values[:,[0,1]]
        #xi = pi.values[:,[0,1]]
        
        pi = (pi["Count"].values/(df[x].size))

        #print("")
        #print(pi)

        return xi, pi

    else:            
        #xi = np.unique(a,axis=0) # unikalne wartości (wiersze)   
        ni = df.groupby([x,y]).size().reset_index(name='Count') # zlicza unikalne wiersze

        #ni = df.value_counts()
        #print("")
        #print(a)
        #print("")

        #print("kolumna Count = ")
        xi = ni.values[:,[0,1]]


        ni = (ni["Count"].values)

        


        return xi, ni


def entropy(x): #wyrzucić tą funkcję (?) - NIE - liczy entropię dla jednej kolmuny
    # x - nazwa kolumny (string)

    #column = data[x]
    #xi = np.unique(data[x])    
    #pi = (data.value_counts(x).values)/data[x].count()

    xi, pi = freq(x, True) 
    



    y = pi.size

    #print("\npi = \n", pi)

    ent = 0

    for i in range(y):        
        ent += pi[i] * math.log((1/pi[i]),2)

    return ent        
    
   
def entropy2(x, y, var_name):

    xi, pi = freq2(x, y, True)
    
    xi1, pi1 = freq(x, True)
    xi2, pi2 = freq(y, True)



    t = np.array(pi)
    t_size = t.size
    
    #print("t = ", t)

    #pi_new_array.append(0)

    #x=np.array(pi)

    t = np.append(t, 0)


    #print("t = ", t)
    #print("")
   
    #print("t 0 = ", t[0])
    #print("")

    #print("t 1 = ", t[1])
    #print("")    

    #print("t = ", t)
    #print("")

   


    pi_x_size = xi1.size
    pi_y_size = xi2.size
    

    #print("pi_y size = ", pi_y_size)
    #print("pi_x size = ", pi_x_size)

    #pi_new_size = pi_x_size * pi_y_size

    #print("pi_new size = ", pi_new_size)

    xxx=0
    while t_size < pi_x_size*pi_y_size:
        #print("t size = ", t_size)
        t = np.append(t, 0)
        t_size = t.size
        #xxx=xxx+1
        #pi_new_array_size = pi_new_array.size

    #print("t = ", t)


    if(var_name == "x"):
        


        p1 = pi[2] + pi[3] # prawdopodobieństwo x == True
        p2 = pi[0] + pi[1] # prawdopodobieństwo x == False

        pi = np.array([p1, p2])        
        #y = pi.size
        y = t.size

        ent = 0

        for i in range(y):      
            if(t[i]==0):
                t[i]=0.0001
            ent += (-1 * t[i] * math.log(t[i],2))
       
        return ent        

    elif(var_name == "y"):

        p1 = pi[1] + pi[3]
        p2 = pi[0] + pi[2]

        pi = np.array([p1, p2])        
        y = pi.size

        ent = 0

        for i in range(y):            
            ent += pi[i] * math.log((1/pi[i]),2)
       
        return ent  

def H_x_y(x, y): # = entropy2
    # x - column1
    # y - column2
    # H_x_y = H(x) + H(y) - I(x,y) 


    xi, pi = freq2(x, y, True)

    #print("xi = \n", xi)
    #print("pi = \n", pi)

    ent = 0

    y = pi.size

    for i in range(y):        
        ent += (pi[i] * math.log((1/pi[i]),2))
    #print("entropia = \n", ent)
    return ent

def H_yIx(x, y):
    xi, pi = freq2(x, y, True)

    #print("xi = ", xi, "\n")
    #print("pi = ", pi, "\n")

    #H = []

    H1 = pi[2] * math.log((1/pi[2]),2) + pi[3] * math.log((1/pi[3]),2) # to się zgadza
    H2 = pi[0] * math.log((1/pi[0]),2) + pi[1] * math.log((1/pi[1]),2) 


    #print("pi = ", pi)
    #print("H1 = ", H1)
    #print("H2 = ", H2)

    #print("\nH=\n")
    #print("H0=", H1)
    #print("H1=", H2)

    H_YX = ((pi[3] + pi[2]) * H1) + ((pi[1] + pi[0]) * H2)  

    #print("H_YX=", H_YX)

    return H_YX

def info_gain(x, y):

    H_x = entropy2(x, y, "x")
    H_y = entropy2(x, y, "y")
    #H_xy = H_x_y(x, y) #entropia wspólna
    H_xy = entropy2(x,y,"x")

    #print("H_x = ", H_x)
    #print("H_y = ", H_y)
    #print("H_xy = ", H_xy)

    I_X_Y = H_x + H_y - H_xy # informacja wzajemna

    #print("I_X_Y = ", I_X_Y)

        #H_YX = H_yIx(x, y)

        #I_Y_X = H_y - H_YX # przyrost informacji
    #print("H_y = ", H_y)
    #print("H_YX = ", H_YX)
    #print("I_Y_X = ", I_Y_X)

    #return I_X_Y, I_Y_X
    return I_X_Y

        


def freq3(x, prob):     
   
    

    if(prob == True):
        print("liczba elementów w kolumnie - hair: ")
        print(data[x].count())

        pi = (data.value_counts(x).values)/data[x].count()

        #print(" pi =  ")
        #print(pi)

        return pi

        
    else:

    # x - numer kolumny ? (0, ... N) ?
    # x - nazwa kolumny (string) ? 
    #x = x.unique()

    # xi, ni / pi

    #value = np.unique(data.iloc[:,1])
        xi = np.unique(data[x])
        ni = data.value_counts(x).values

       
    #print(" q = ", q)

        print(" ")
        print(" xi = ", xi)

        print(" ")
        print(" ni = ", ni)

       



    #print(" x2 = ", x2)

    

    #print(" np unique = ")
    #print(np.unique(data[x]))
    #print(" = np unique  ")
    

        return xi, ni
    # niech funkcja wyświetli kolumnę o nazwie opdanej w x :


def freq11(x, prob):     
   
    xi = x.unique()
    ni = x[x==prob].value_counts()
    return xi, ni

#data = pn.read_csv('zoo.csv') 
#print(data)
#print("pierwsza kolumna:")

#print(data.iloc[:,0] ) 
#print(" ")

#print(data.value_counts('hair'))
#print(" ")

#z, q = data.value_counts('hair').values

#print(" ")
#print(" z = ", z)
#print(" q = ", q)

#print(data.count())
#print(" ")

#print(data.loc[data['hair'] == 'True'])
#print(" ")

##print(data.unique())

##x = np.unique(data.iloc[:,1]).count # zwraca unikalne wartości w danej kolumnie
##print(x)

#print("moja kolumna = ")

#v = "hair"

#print(data[v])

#print("")
#print("efekt działania funkcji =")
##print(freq("hair",True)[0])
##print(freq("hair",True)[1])

##print(freq("hair",False))
#xi, ni = freq("hair",True)
#print("xi = ", xi)
#print("ni = ", ni)

#print("")
#print("efekt działania funkcji freq2=")
##print(freq("hair",True)[0])
##print(freq("hair",True)[1])

##a = np.array([[1,2,3,4,5], ['a', 'b', 'a', 'b', 'b']])
##a = np.transpose(a)
##print(a)

## Nieco zmodyfikowałem polecenie :
## df = pd.DataFrame(”x”: [1, 2, 3, 4, 5], ”y”: [’a’, ’b’, ’a’, ’b’, ’b’], index=np.arange(5))
## ponieważ nie chciało mi ono działać, zamiast tego stworzyłem zamierzony obiekt data frame w inny sposób :

##df = pn.DataFrame(a, index=randint(1,10,5), columns=['X','Y'])

#hair1 = data["hair"]
#eggs1 = data["eggs"]

##hair1 = hair1.values
#print("kolumna hair =")
#print(data["hair"].values)

#print("kolumna eggs =")
#print(data["eggs"].values)


#a = np.array([data["tail"].values, data["eggs"].values])
##a = np.reshape(a, (2,100))
#print("a = \n")
#print(a)

#print("a size = ", a.size)

#a = np.transpose(a)

#print("a po zamianie = ")
#print(a)

#b = np.array([[1,2],[2,2],[1,2],[1,2],[2,2],[5,3]])
#print("b = \n")
#print(b)



#print("unikalne wartości w a = ")
#print(np.unique(a,axis=0))
#print("")

#df = pn.DataFrame(a, columns=['hair','eggs'])
#print("df = ")
#print(df)

#print("########################################")
#data_x = pn.read_csv('zoo.csv') 
#print("hair")
#print(data_x['hair'].values) 

#xxx = (data_x['hair'].values) 

#my_array = np.array(xxx)
##b = np.resize(xxx,(101,1)) 

#print("xxx = ")
#print(xxx)

#print("my_array = ")
#print(my_array)

#b = np.resize(my_array,(101,2))

#print("b = ")
#print(b)

#print("########################################")


#print("value counts = ")
#print(df.value_counts().values)
#print("")

#df = pn.DataFrame(b, columns=['x','y'])
#print("df = ")
#print(df)

#print("value counts = ")
#print(df.value_counts().values) # tutaj skończyłem - zliczanie unikalnych wierszy w data frame

#print("###########################################################")

#print(freq2("hair","eggs",True))

#print("###########################################################")
#print("###########################################################")

#print(entropy("hair"))

#print("###########################################################")




#a = np.array([[1,2,3,4,5], ['a', 'b', 'a', 'b', 'b']])
#a = np.resize(a, (5,2))
#print("a po zamianie = ")
#print(a)

#c = np.array([["True","False"],["True","False"],["False","True"],["False","False"],["True","True"],["False","True"]])
#print("c = \n")
#print(c)

#df = pn.DataFrame(c, columns=['x','y'])
#print("df = ")
#print(df)

#print("value counts = ")
#print(df.value_counts().values) 
#print("")


#print(freq2("hair", "eggs", False))
#xi, ni = freq("hair",True)
#print("xi = ", xi)
#print("ni = ", ni)

#f = "hair"
#print(data.loc[data[f]])

#print("")
#print("efekt działania funkcji freq1=")
#hair = data["hair"]
#print("hair = ")
#print(hair)
#print("\n")
#print(freq1(hair,False))

data = pn.read_csv('zoo.csv') 
df = pn.DataFrame({'x':data["hair"],'y':data["type"]})
print("\n df = \n")
print(df)


#print("##############################################################################################################")
print("")
print("efekt działania funkcji freq = \n")

xi, pi = freq("hair", True)
print(xi)
print(pi)

xi, ni = freq("hair", False)
print(xi)
print(ni)

print("")
print("efekt działania funkcji freq2 = \n")

xi, pi = freq2("hair", "eggs", True)
print(xi)
print(pi)
#xi, pi = freq2("hair", "eggs", True)
#print(xi, "\n", pi)

print("###########################################################")


#xi, ni = freq2("hair", "eggs", False)
#print(xi, "\n", ni)

#print(freq2("hair", "eggs", False))


#print("")
#print("efekt działania funkcji entropy = \n")
#print("")

#ent = entropy("hair")
#print("entropia = ", ent)


#print("")
#print("efekt działania funkcji entropy2 = \n")
#print("")
#print(entropy2("hair", "eggs", "x"))
#print("")
#print(entropy2("hair", "eggs", "y"))

#print("")
#print("efekt działania funkcji H_x_y = \n")
#print("")

#print(H_x_y("hair","eggs"))

#print("")
#print("efekt działania funkcji H_yIx = \n")
#print(H_yIx("hair","eggs"))


#print("")
#print("efekt działania funkcji info_grain = \n")
#print("")

#IW, IG = info_gain("hair","eggs")
#print("informacja wzajemna =", IW, "\nPrzyrost informacji = ", IG) 


print("")
print("efekt działania funkcji freq = \n")
print("")

xi, pi = freq("type", True)
print(xi, pi)

#print("")
#print("efekt działania funkcji freq2 = \n")

#x1, p1 = freq2("hair", "type", False)
#print(x1, "\n", p1)

#print("\n df - unique  = \n")
#print(df['x'].unique())

#print("---------------------------------------")
#print("---------------------------------------")
##df.drop_duplicates()
#print(df.drop_duplicates().values)

#print("---------------------------------------")
#print("---------------------------------------")
#df = df.groupby(['y','x']).size()
#df = df.groupby(['y','x']).size()

#print("\n df = ")
#print(df)
#print("\n")


#print("")
#print(df.groupby(['y','x']).size())
#print(df.values[:,0])
#print("")

#print(df.iloc[:,1:2])


#df = df.drop_duplicates()
#print("\ndf drop_duplicates = \n")
#print(df.values)


print("#########################################")
print("---------------------------------------")
print("efekt działania funkcji freq2 = \n")
xi, pi = freq2("hair", "type", True)
print(xi)
print(pi)
print("---------------------------------------")
print("#########################################")
print("------------------------------------------------------------------------------------------------")
print("test")
data = pn.read_csv('zoo.csv') 
df = pn.DataFrame({'x':data["hair"],'y':data["type"]})
print("\n df = \n")
print(df)

print("---------------------------------------")
print("#########################################")
print("\n efekt działania funkcji freq1 = ")

print("")
xi, pi = freq("type", True)
print(xi)
print(pi)

print("---------------------------------------")
print("#########################################")


print("---------------------------------------")
print("#########################################")
print("\n efekt działania funkcji freq2 = ")

print("")
xi, pi = freq2("eggs", "type", True)
print(xi)
print(pi)

print("")

print("")
xi, ni = freq2("hair", "type", False)
print(xi)
print(ni)
print("")
print("---------------------------------------")
print("#########################################")


print("\n efekt działania funkcji entropy = ")

print("")
print(entropy("type"))
print("")

xi, pi = freq( "type", True)
print(xi)
print(ni)
print("")

print("---------------------------------------")
print("#########################################")
print("\n efekt działania funkcji entropy2= ")

#print(entropy2("hair", "type", True))
ent = entropy2("hair", "type", "x")

print("ENTROPIA = ", ent)

print("---------------------------------------")
print("#########################################")

print("###################################################################################")
data = pn.read_csv('zoo.csv') 
df = pn.DataFrame({'x':data["hair"],'y':data["type"]})




a = {'x': ['True', 'True',  'True', 'True', 'True',  'True', 'True', 'False', 'False', 'False','False', 'False', 'False', 'False', ], 'y': ['amphibian', 'bird', 'fish', 'insect', 'invertebrate', 'mammal', 'reptile', 'amphibian', 'bird', 'fish', 'insect', 'invertebrate', 'mammal', 'reptile']}
df1 = pn.DataFrame(a) # utworzy obiekt data frame ze zmiennej słownik
print("\ndf1=")
print(df1)

print("\ndf1=")
print(df1.values[0,0])

y = df1['x'].size

#t = np.array(string,shape=(9,2))
#print("")
#t[0,0]="True"




print("#########################################")
print("\n efekt działania funkcji freq2= ")

print("")
xi, pi = freq2("legs", "type", True)
print(xi)
print(pi)


print("xi 0 = ", xi[0,0])





#pi = df.groupby([x,y]).size().reset_index(name='Count')         
#xi = pi.values[:,[0,1]]

print("---------------------------------------")
print("#########################################")
print("\n efekt działania funkcji info gain= ")

IG1 = info_gain("hair", "type")
IG2 = info_gain("feathers", "type")
IG3 = info_gain("eggs", "type")
IG4 = info_gain("milk", "type")
IG5 = info_gain("airborne", "type")
IG6 = info_gain("aquatic", "type")
IG7 = info_gain("predator", "type")
IG8 = info_gain("toothed", "type")
IG9 = info_gain("backbone", "type")
IG10 = info_gain("breathes", "type")
IG11 = info_gain("venomous", "type")
IG12 = info_gain("fins", "type")
IG13 = info_gain("legs", "type")
IG14 = info_gain("tail", "type")
IG15 = info_gain("domestic", "type")
IG16 = info_gain("catsize", "type")
print("IG = ", IG1)
print("IG = ", IG2)
print("IG = ", IG3)
print("IG = ", IG4)
print("IG = ", IG5)
print("IG = ", IG6)
print("IG = ", IG7)
print("IG = ", IG8)
print("IG = ", IG9)
print("IG = ", IG10)
print("IG = ", IG11)
print("IG = ", IG12)
print("IG = ", IG13)
print("IG = ", IG14)
print("IG = ", IG15)
print("IG = ", IG16)

a = np.array([IG1, IG2, IG3, IG4, IG5, IG6, IG7, IG8, IG9, IG10, IG11, IG12, IG13, IG14, IG15, IG16])
print("a = ", a)
print("")

print(np.sort(a))


print("---------------------------------------")
print("#########################################")

