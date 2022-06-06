
# przyporządkowanie próbki (punktu) do danej grupy na podstawie sąsiadów

# obliczanie odległości euklidesowej

import numpy as np
from collections import Counter



def euclidean_distance(x1, x2):    
    # odległość euklidesowa dla dwóch wektorów cech
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, n_neighbors=3):  
        self.n_neighbors = n_neighbors

    def fit(self, X, y):      

        self.X_train = X 
        self.y_train = y 

    def predict(self, X):      

        predicted = [self._predict(x) for x in X] 

        #predicted_labels = []

        #for row in X:
        #    label = self.closest(row)
        #    predicted_labels.append(label)

        #return predicted_labels

        return np.array(predicted)
       
    def _predict(self, x):    
           
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]             
            
        n_indexes  = np.argsort(distances)[:self.n_neighbors] 

        n_nearest_lables = [self.y_train[i] for i in n_indexes]       

        most_common = Counter(n_nearest_lables).most_common(1)

        #best_dist = distance.euclidean(row, self.features_train[0])
        #best_index = 0

        #return most_common[:0]
        # most_common[0]
        # most_common[1]
        return most_common[0][0]

    def score(self, x, y):
        #predictions = clf.predict(y)
        predictions = self.predict(y)

        acc = np.sum(predictions == x) / len(x)
        print("\nwskaźnik jakości dopasowania =", acc * 100, "\n")

        