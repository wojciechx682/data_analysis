import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import cv2
from PIL import Image
from math import sqrt
PI = math.pi

def quantization(x, bins):     
    
    # Funkcja dokonuje kwantyzacji obrazu na podstawie tablicy bins wygenerowanej przez histogram, zwraca nową wartość dla każdego piksela - będącą środkiem przediału do którego należy wartość tego piksela
    # x - wartość piksela
    # bins - tablica bins zwrócona przez histogram

    i = 0   
    if (x == bins[-1]):
        return (bins[-1]+bins[-2])/2
    elif (x == bins[0]):
        return (bins[0]+bins[1])/2
    else:
        while x > bins[i]:    
            i=i+1
        return (bins[i]+bins[i-1])/2

def binarization(x, bins):
    
    # Funkcja dokonuje binaryzacji obrazu na podstawie progu, będącego minimum z wartości funkcji histogramu
    # wartości pikseli będące mniejsze od tego progu, zamienia na wartość 0,
    # wartości pikseli będące większe od tego progu - zamienia na wartość 1
    # x - wartość piksela   
    # bins - tablica bins zwrócona przez funkcję histogramu       
    
    bins_min = bins[1:-1].min()

    if(x<bins_min):
        return 0
    else:
        return 1
     
def discretization(f, Fs):

    dt = (1/Fs)
    t = np.array(np.arange(0,1,dt))  
    s = []
    y = t.size

    for x in range(y):
        s += [math.sin(2*PI*f*t[x])]

    return t, s
   

x1 = discretization(10, 20)
x2 = discretization(10, 21)
x3 = discretization(10, 30)
x4 = discretization(10, 45)
x5 = discretization(10, 50)
x6 = discretization(10, 100)
x7 = discretization(10, 150)
x8 = discretization(10, 200)
x9 = discretization(10, 250)
x10 = discretization(10, 1000)

plt.plot(x1[0],x1[1])
plt.show()
plt.plot(x2[0],x2[1])
plt.show()
plt.plot(x3[0],x3[1])
plt.show()
plt.plot(x4[0],x4[1])
plt.show()
plt.plot(x5[0],x5[1])
plt.show()
plt.plot(x6[0],x6[1])
plt.show()
plt.plot(x7[0],x7[1])
plt.show()
plt.plot(x8[0],x8[1])
plt.show()
plt.plot(x9[0],x9[1])
plt.show()
plt.plot(x10[0],x10[1])
plt.show()

# Czy istnieje twierdzenie, które określa z jaką częstotliwością należy próbkować, aby móc wiernie odtworzyć sygnał? Jak się nazywa?
# Odp.: Tak, jest to Twierdzenie o próbkowaniu (twierdzenie Nyquista–Shannona) : mówi ono o tym, że aby móc wierne odtworzyć sygnał cyfrowy z sygnału analogowego częstotliwość próbkowania powinna być conajmniej dwa razy większa od częstotliwości sygnału.

# Jak nazywa się zjawisko, które z powodu błędnie dobranej częstotliwości próbkowania powoduje błędną interpretację sygnału?
# Jest to Aliasing - wygenerowany zostaje inny sygnał niż sygnał wejściowy, ze względu nie spełnienia twierdzenia o próbkowaniu (gdy częstotliwość próbkowania jest mniejsza niż częstotliwość sygnału - próbki "nienadążają" za częstotliwością sygnału - są rozmieszczone w zbyt dużych odstępach czasowych).

img = mpimg.imread('d.png')

methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

np.random.seed(19680801)
grid = np.random.rand(4, 4)
fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})

for ax, interp_method in zip(axs.flat, methods):
    ax.imshow(img, interpolation=interp_method, cmap='viridis')
    ax.set_title(str(interp_method))

plt.tight_layout()
plt.show()

# KWANTYZACJA

img = mpimg.imread('d.png')
plt.imshow(img)
plt.show()

# Wykonaj polecenie, które zwróci ile wymiarów ma wczytana macierz (obrazek):
print(len(img.shape))
#print(np.ndim(img))


# Wykonaj polecenie, które zwróci iloma wartościami jest opisywany pojedynczy piksel (inaczej: z ilu wartości składa się najgłębszy wymiar):
print(img.shape[2])
#print(img.shape)
#print(np.size(img[1,1,:]))
#print(img[:,:,1]) 
#print(np.max(img))

# Przekształć obraz do skali szarości za pomocą 3 różnych metod (zapisz jako 3 różne macierze):

gray_image1 = np.copy(img)
gray_image2 = np.copy(img)
gray_image3 = np.copy(img)
img_test = np.copy(img)

x_width = np.size(img[1,:,0]) # "Wszystkie kolumny dla 1-go wiersza -> czyli zwróci ilość wierszy (szerokość)
print("image_width = ", x_width)

y_height = np.size(img[:,1,0]) # "Wszystkie wiersze dla 1-wszej kolumny wiersza -> czyli zwróci ilość kolumn (wysokość)
print("img_height = ", y_height)

z_depth = np.size(img[1,1,:]) # Ilość składowych opisujących kolor: R(0), G(1), B(2), A(3)
print("img_rgba = ", z_depth)

print("img_rgba = ", gray_image1[0,0,0:4])



for x in range(x_width):
    for y in range(y_height):
        #for z in range(z_depth-1):
            gray_image1[y,x,0:4] = (np.max(img[y,x,0:4]) + np.min(img[y,x,0:4]))/2
#plt.imshow(gray_image1)
#plt.show()

for x in range(x_width):
    for y in range(y_height):
        #for z in range(z_depth-1):
            gray_image2[y,x,0:4] = (img[y,x,0] + img[y,x,1] + img[y,x,2])/3
#plt.imshow(gray_image2)
#plt.show()

for x in range(x_width):
    for y in range(y_height):
        #for z in range(z_depth-1):
            gray_image3[y,x,0:4] = (0.21*img[y,x,0] + 0.72*img[y,x,1] + 0.07*img[y,x,2])
#plt.imshow(gray_image3)
#plt.show()

print("HISTOGRAMY:")

histogram, bin_edges = np.histogram(gray_image1, bins=16, range=(0,1))
plt.figure()
plt.title("gray_image1 - histogram")
plt.xlabel("wartość piksela")
plt.ylabel("ilość pikseli")
plt.xlim([0.0, 1.0])
plt.plot(bin_edges[0:-1], histogram)
plt.show()

histogram, bin_edges = np.histogram(gray_image2, bins=16, range=(0,1))
plt.figure()
plt.title("gray_image2 - histogram")
plt.xlabel("wartość piksela")
plt.ylabel("ilość pikseli")
plt.xlim([0.0, 1.0])
plt.plot(bin_edges[0:-1], histogram)
plt.show()

histogram, bin_edges = np.histogram(gray_image3, bins=16, range=(0,1))
plt.figure()
plt.title("gray_image3 - histogram")
plt.xlabel("wartość piksela")
plt.ylabel("ilość pikseli")
plt.xlim([0.0, 1.0])
plt.plot(bin_edges[0:-1], histogram)
plt.show()

img_hist = np.ravel(img)
gray_img1_hist = np.ravel(gray_image1)
gray_img2_hist = np.ravel(gray_image2)
gray_img3_hist = np.ravel(gray_image3)

# Wygeneruj histogram dla każdego z otrzymanych „szarych” obrazów:
# histogramy wygenerowały ze zredukowaną liczbą kolorów do 16 (bins = 16):

(n, bins, patches) = plt.hist(img_hist, bins=3)
plt.show()
print("\nZAKRESY NOWYCH KOLORÓW (BINS): ")
print(bins)

(n1, bins1, patches1) = plt.hist(gray_img1_hist, bins=16)
plt.show()
print("\nZAKRESY NOWYCH KOLORÓW (BINS1): ")
print(bins1)

(n2, bins2, patches2) = plt.hist(gray_img2_hist, bins=16)
plt.show()
print("\nZAKRESY NOWYCH KOLORÓW (BINS2): ")
print(bins2)

(n3, bins3, patches3) = plt.hist(gray_img3_hist, bins=16)
plt.show()
print("\nZAKRESY NOWYCH KOLORÓW (BINS3): ")
print(bins3)

plt.imshow(img)
plt.show()

plt.imshow(gray_image1)
plt.show()

plt.imshow(gray_image2)
plt.show()

plt.imshow(gray_image3)
plt.show()

print("----------------------------------------------------")
print(" ZAMIANA OBRAZKA 'img.png' - REDUKCJA KOLORÓW: ")
print(" przed zamianą = ")
plt.imshow(img_test)
plt.show()

print("kwantyzacja ... ")
for x in range(x_width):
    for y in range(y_height):
        for z in range(3):            
            img_test[y,x,z] = quantization(img_test[y,x,z],bins)

print("po kwantyzacji :")
plt.imshow(img_test)
plt.show()

print("\n\nBINARYZACJA\n\n")

img = mpimg.imread('bin3.png')
plt.imshow(img)
plt.show()

gray_image_bin = np.copy(img)

x_width = np.size(img[1,:,0]) # "Wszystkie kolumny dla 1-go wiersza -> czyli zwróci ilość wierszy (szerokość)
print("image_width = ")
print(x_width) 

y_height = np.size(img[:,1,0]) # "Wszystkie wiersze dla 1-wszej kolumny wiersza -> czyli zwróci ilość kolumn (wysokość)
print("img_height = ")
print(y_height) 

z_depth = np.size(img[1,1,:]) # Ilość składowych opisujących kolor: R(0), G(1), B(2), A(3)
print("img_rgba = ")
print(z_depth) #4

for x in range(x_width):
    for y in range(y_height):
        #for z in range(3):
            gray_image_bin[y,x,0:4] = (np.max(img[y,x,0:4])+np.min(img[y,x,0:4]))/2
plt.imshow(gray_image_bin)
plt.show()

gray_image_bin_hist = np.ravel(gray_image_bin)

(n, bins, patches) = plt.hist(gray_image_bin_hist, bins=256)
plt.show()

print("bins: ", bins)
print("n: ", n)

histogram, bin_edges = np.histogram(gray_image_bin*gray_image_bin, bins='auto', range=(0,1)) # podniesienie wartości macierzy do kwadratu, aby "wzmocnic" różnice w wartościach kolorów na histogramie
plt.figure()
plt.title("histogram")
plt.xlabel("wartość piksela")
plt.ylabel("ilość pikseli")
plt.xlim([0.0, 1.0])
plt.plot(bin_edges[0:-1], histogram)
plt.show()

print("bins . min = ", bins[1:-1].min())
print("n . min = ", n[1:-1].min())

for x in range(x_width):
    for y in range(y_height):
        for z in range(3):
            gray_image_bin[y,x,z] = binarization(gray_image_bin[y,x,z], bins)
plt.imshow(gray_image_bin)
plt.show()