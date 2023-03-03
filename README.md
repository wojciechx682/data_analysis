# data_analysis

pca.py - principal component analysis, PCA

  The code performs the following steps:

    Imports the necessary libraries and packages such as numpy, pandas, scipy, and scikit-learn.
    Defines a function draw_vector() that is used later to plot vectors.
    Generates a 2D random dataset consisting of 200 samples by multiplying two randomly generated 2x2 matrices.
    Visualizes the randomly generated dataset using the matplotlib scatter plot function.
    Uses the scikit-learn package
    Implements wiPCA() 
    Computes and visualizes wiPCA() function
    
numpy.py
  
    The code is a Python program that demonstrates the use of NumPy, a library for numerical computing in Python. It imports NumPy and random, and then creates and manipulates various arrays and matrices using NumPy functions and methods.
    
pandas.py
  
    The code demonstrates some basic functions of Pandas, such as mean(), concat(), and sort_values(). 
    
k_means.py

    This code performs the k-means clustering algorithm. It imports several libraries including numpy, pandas, scipy, matplotlib, math, decimal, and sklearn. The code defines a function ksrodki(X, k_number, a) which takes a 2D data array X, the number of clusters k_number, and a boolean value a that determines whether to show the plotted output.

    The function first initializes k_number centroids randomly and calculates the Euclidean distance between each point in X and the centroids. It assigns each point to the closest centroid, forms k clusters, and computes the new centroid of each cluster. This process is repeated several times (15 times in this code).

    The code then plots the clusters using matplotlib, with each cluster represented by a different color. The resulting plot is either shown or not depending on the value of the boolean parameter








