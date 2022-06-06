
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from sklearn import datasets
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from matplotlib.colors import ListedColormap
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import timeit
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

print("\n Hello ! \n")

a = np.array(([1,2,3],[4,5,6]))

print("\na = \n", a)

X, y = datasets.make_classification(
    n_samples = 100,
    n_features = 2,
    n_classes = 2,
    n_informative = 2,
    n_clusters_per_class=2,
    n_redundant = 0,
    n_repeated = 0,  
    #random_state = 7    
)

print("\nX = \n", X)

print("\ny = \n", y)
print("\nsize of y =", len(y))

colors = []

for i in range(len(y)):
    if(y[i]==0):        
        colors += ["#9d0141"]
    else:
        colors += ["#5e4fa2"]

plt.scatter(X[:, 0], X[:, 1], color=colors)
plt.title('random data,  n_classes = 2')
plt.show()

X1, y1 = np.arange(10).reshape((5, 2)), range(5)

print("\nX1 = \n", X1)
print("\ny1 = \n", list(y1))

#X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print("\nX_train = \n", X_train)
print("\ny_train = \n", y_train)
print("\nX_test = \n", X_test)
print("\ny_test = \n", y_test)

clf_ = [GaussianNB(), QuadraticDiscriminantAnalysis(), KNeighborsClassifier(n_neighbors=3), svm.SVC(probability=True), DecisionTreeClassifier()]

print("\nclf[0] tablica = ", clf_[0])

acc_arr1 = []
acc_arr2 = []
acc_arr3 = []
acc_arr4 = []
acc_arr5 = []

arr_sns1 = []
arr_sns2 = []
arr_sns3 = []
arr_sns4 = []
arr_sns5 = []

arr_prec1 = []
arr_prec2 = []
arr_prec3 = []
arr_prec4 = []
arr_prec5 = []

arr_F1_1 = []
arr_F1_2 = []
arr_F1_3 = []
arr_F1_4 = []
arr_F1_5 = []

arr_auc1 = []
arr_auc2 = []
arr_auc3 = []
arr_auc4 = []
arr_auc5 = []

arr_tran_t1 = []
arr_tran_t2 = []
arr_tran_t3 = []
arr_tran_t4 = []
arr_tran_t5 = []

arr_test_t1 = []
arr_test_t2 = []
arr_test_t3 = []
arr_test_t4 = []
arr_test_t5 = []


arr_acc = [acc_arr1, acc_arr2, acc_arr3, acc_arr4, acc_arr5]
arr_sns = [arr_sns1, arr_sns2, arr_sns3, arr_sns4, arr_sns5]
arr_prec = [arr_prec1, arr_prec2, arr_prec3, arr_prec4, arr_prec5]
arr_F1 = [arr_F1_1, arr_F1_2, arr_F1_3, arr_F1_4, arr_F1_5]
arr_auc = [arr_auc1, arr_auc2, arr_auc3, arr_auc4, arr_auc5]
arr_train_t = [arr_tran_t1, arr_tran_t2, arr_tran_t3, arr_tran_t4, arr_tran_t5]
arr_test_t = [arr_test_t1, arr_test_t2, arr_test_t3, arr_test_t4, arr_test_t5]

for j in range(100):

    for i in range(5):        

        rand_number = random.randint(0,50)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=rand_number)

        clf = clf_[i]   

        start1 = timeit.default_timer()

        fit_ = clf.fit(X_train, y_train)

        stop1 = timeit.default_timer()

        train_time = stop1-start1
       

        start2 = timeit.default_timer()

        y_pred = clf.predict(X_test)

        stop2 = timeit.default_timer()

        test_time = stop2-start2       

        acc = accuracy_score(y_test, y_pred)
        sns = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)

        arr = arr_acc[i]
        arr1 = arr_sns[i]
        arr2 = arr_prec[i]
        arr3 = arr_F1[i]
        arr4 = arr_auc[i]
        arr5 = arr_train_t[i]
        arr6 = arr_test_t[i]

        arr += [acc]
        arr1 += [sns]
        arr2 += [prec]
        arr3 += [f1]
        arr4 += [roc]
        arr5 += [train_time]
        arr6 += [test_time]
    
    if(j==99 and i==4):

        clf = clf_[0]
        clf.fit(X, y)
        y_pred = clf.predict(X)

        colors_ = []

        for i in range(len(y_pred)):   
            if(y_pred[i]==0):        
                colors_ += ["#9d0141"]
            else:
                colors_ += ["#5e4fa2"]

        colors_diff = []

        for i in range(len(y)):   
            if(y_pred[i]==y[i]):        
                colors_diff += ["#006837"]
            else:
                colors_diff += ["#a50026"]    

     

print("\nacc_arr1 = ", acc_arr1, "\n")
print("\arr_sns1 = ", arr_sns1, "\n")

print("\nacc_arr2 = ", acc_arr2, "\n")
print("\narr_sns2 = ", arr_sns2, "\n")


print("\nacc_arr5 = ", acc_arr5, "\n")
print("\narr_sns5 = ", arr_sns5, "\n")

c1_acc_mean = np.mean(acc_arr1)
c2_acc_mean = np.mean(acc_arr2)
c3_acc_mean = np.mean(acc_arr3)
c4_acc_mean = np.mean(acc_arr4)
c5_acc_mean = np.mean(acc_arr5)

c1_sns_mean = np.mean(arr_sns1)
c2_sns_mean = np.mean(arr_sns2)
c3_sns_mean = np.mean(arr_sns3)
c4_sns_mean = np.mean(arr_sns4)
c5_sns_mean = np.mean(arr_sns5)

c1_prec_mean = np.mean(arr_prec1)
c2_prec_mean = np.mean(arr_prec2)
c3_prec_mean = np.mean(arr_prec3)
c4_prec_mean = np.mean(arr_prec4)
c5_prec_mean = np.mean(arr_prec5)

c1_F1_mean = np.mean(arr_F1_1)
c2_F1_mean = np.mean(arr_F1_2)
c3_F1_mean = np.mean(arr_F1_3)
c4_F1_mean = np.mean(arr_F1_4)
c5_F1_mean = np.mean(arr_F1_5)

c1_auc_mean = np.mean(arr_auc1)
c2_auc_mean = np.mean(arr_auc2)
c3_auc_mean = np.mean(arr_auc3)
c4_auc_mean = np.mean(arr_auc4)
c5_auc_mean = np.mean(arr_auc5)

c1_train_mean = np.mean(arr_tran_t1) * 200
c2_train_mean = np.mean(arr_tran_t2) * 200
c3_train_mean = np.mean(arr_tran_t3) * 200
c4_train_mean = np.mean(arr_tran_t4) * 200
c5_train_mean = np.mean(arr_tran_t5) * 200

c1_test_mean = np.mean(arr_test_t1) * 200
c2_test_mean = np.mean(arr_test_t2) * 200
c3_test_mean = np.mean(arr_test_t3) * 200
c4_test_mean = np.mean(arr_test_t4) * 200 
c5_test_mean = np.mean(arr_test_t5) * 200


print("\nc1_acc_mean = ", c1_acc_mean, "\n")
print("\nc2_acc_mean = ", c2_acc_mean, "\n")
print("\nc3_acc_mean = ", c3_acc_mean, "\n")
print("\nc4_acc_mean= ", c4_acc_mean, "\n")
print("\nc5_acc_mean = ", c5_acc_mean, "\n")

print("\nc1_acc_mean = ", c1_sns_mean, "\n")
print("\nc2_acc_mean = ", c2_sns_mean, "\n")
print("\nc3_sns_mean = ", c3_sns_mean, "\n")
print("\nc4_sns_mean = ", c4_sns_mean, "\n")
print("\nc5_sns_mean = ", c5_sns_mean, "\n")


print("\nc1_test_mean = ", c1_test_mean, "\n")
print("\nc2_test_mean = ", c2_test_mean, "\n")
print("\nc3_test_mean = ", c3_test_mean, "\n")
print("\nc4_test_mean= ", c4_test_mean, "\n")
print("\nc5_test_mean = ", c5_test_mean, "\n")

plotdata = pd.DataFrame({"GaussianNB":[c1_acc_mean, c1_sns_mean, c1_prec_mean, c1_F1_mean, c1_auc_mean, c1_train_mean, c1_test_mean],"QuadraticDiscriminant":[c2_acc_mean, c2_sns_mean, c2_prec_mean, c2_F1_mean, c2_auc_mean, c2_train_mean, c2_test_mean],"KNeighbors":[c3_acc_mean, c3_sns_mean, c3_prec_mean, c3_F1_mean, c3_auc_mean, c3_train_mean, c3_test_mean],"svm.SVC()":[c4_acc_mean, c4_sns_mean, c4_prec_mean, c4_F1_mean, c4_auc_mean, c4_train_mean, c4_test_mean],"DecisionTree":[c5_acc_mean, c5_sns_mean, c5_prec_mean, c5_F1_mean, c5_auc_mean, c5_train_mean, c5_test_mean]}, index=["acc", "rec", "prec", "f1", "roc_auc","train_t","test_t"])


plotdata.plot(kind="bar")
plt.show()

print("\n\n -------------> PETLA \n\n")
g=0
for i in range(len(clf_)):
    
    clf__ = clf_[i]        
    clf__.fit(X, y)        
    y_pred = clf__.predict(X)        
    colors_ = []
        
    for i in range(len(y_pred)):   
            
        if(y_pred[i]==0):        
                
            colors_ += ["#9d0141"]              
        else:                
            colors_ += ["#5e4fa2"]
            colors_diff = []

    for i in range(len(y)):   
        if(y_pred[i]==y[i]):        
            colors_diff += ["#006837"]
        else:
            colors_diff += ["#a50026"] 
            
    fig, axs = plt.subplots(1, 3)    
    fig.suptitle(clf_[g])
    axs[0].scatter(X[:, 0], X[:, 1], color=colors) 
    axs[0].set_title('oczekiwane')
    axs[1].scatter(X[:, 0], X[:, 1], color=colors_)   
    axs[1].set_title('obliczone')
    axs[2].scatter(X[:, 0], X[:, 1], color=colors_diff)
    axs[2].set_title('różnice')
    g=g+1
    plt.show()

for i in range(len(clf_)):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    
    model = clf_[i]
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)
   
    probs = probs[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.3f' % auc)
   
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    
    plt.plot([0, 1], [0, 1], linestyle='--')
   
    lines = plt.plot(fpr, tpr, marker='.')
    
    plt.title(clf_[i])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(lines[:2], ['AUC = %.3f' % auc, 'second']);
    plt.show()

for i in range(len(clf_)):

    clf =  clf_[i]

    h = .02  
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  
    Z = Z.reshape(xx.shape)    
    plt.figure()   
    plt.pcolormesh(xx, yy, Z, shading='auto', cmap=cmap_light)

     # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

##########################################################################################################

#print("\n\n TEST - KLASYFIKATOR - Rysunek 3 :")
#clf = GaussianNB()

print("\n ZADANIE 2 - badanie parametrów wybranego klasyfikatora")

X1, y1 = datasets.make_classification(
    n_samples = 200,
    n_features = 2,
    n_classes = 2,
    n_informative = 2,
    n_clusters_per_class=2,
    n_redundant = 0,
    n_repeated = 0    
)

colors_ = []

for i in range(len(y1)):
    if(y1[i]==0):        
        colors_ += ["#9d0141"]
    else:
        colors_ += ["#5e4fa2"]

plt.scatter(X1[:, 0], X1[:, 1], color=colors_)
plt.title('random data,  n_classes = 2')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.5, random_state=2)


grid_params = {
    'n_neighbors': list(range(1,50,1)),
    #'weights': ['uniform', 'distance'],
    #'metric':['euclidean','manhattan']
}

gs = GridSearchCV(
    KNeighborsClassifier(),
    grid_params,
    verbose = 1,
    cv = 3,
    n_jobs = -1
)

gs_results = gs.fit(X1, y1)

print("\n---->", gs_results.best_score_)
print("\n---->", gs_results.best_estimator_)
print("\n---->", gs_results.best_params_)


print("\n TEST : \n ")

k_list = list(range(1,50,2))

print("\nk_list = ", k_list)

cv_scores = []



for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

plt.figure()
#plt.figure(figsize=(15,10))
plt.title('the optimal number of neighbors')
plt.xlabel('number of neighbors K')
plt.ylabel('misclassification Error')

plt.plot(k_list, MSE)

plt.show()

########################

neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


#X, y = datasets.make_classification(
#    n_samples = 100,
#    n_features = 2,
#    n_classes = 2,
#    n_informative = 2,
#    n_clusters_per_class=2,
#    n_redundant = 0,
#    n_repeated = 0,  
#    #random_state = 7    
#)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)   

# Generate plot
plt.title('k-NN: Different number of neighbour')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#fig, axs = plt.subplots(2)     
#axs[0].plot(k_list, MSE) 


#axs[0].xlabel('Number of Neighbors')
#axs[0].ylabel('Accuracy')
##axs[0].set_title('oczekiwane')
#axs[1].plot(neighbors, train_accuracy)   
##axs[1].set_title('obliczone')

#plt.show()

#####################################################################

clf = gs_results.best_estimator_

print("\n clf = ", clf)







for j in range(100):

           
    rand_number = random.randint(0,50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=rand_number)

     

    start1 = timeit.default_timer()

    fit_ = clf.fit(X_train, y_train)

    stop1 = timeit.default_timer()

    train_time = stop1-start1     

    start2 = timeit.default_timer()

    y_pred = clf.predict(X_test)

    stop2 = timeit.default_timer()

    test_time = stop2-start2       

    acc = accuracy_score(y_test, y_pred)

    sns = recall_score(y_test, y_pred)

    prec = precision_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc = roc_auc_score(y_test, y_pred)

    arr = arr_acc[0]

    arr1 = arr_sns[0]

    arr2 = arr_prec[0]

    arr3 = arr_F1[0]

    arr4 = arr_auc[0]

    arr5 = arr_train_t[0]

    arr6 = arr_test_t[0]

    arr += [acc]

    arr1 += [sns]

    arr2 += [prec]

    arr3 += [f1]

    arr4 += [roc]

    arr5 += [train_time]

    arr6 += [test_time]
    
    if(j==99):      
        clf.fit(X, y)
        y_pred = clf.predict(X)

        colors_ = []

        for i in range(len(y_pred)):   
            if(y_pred[i]==0):        
                colors_ += ["#9d0141"]
            else:
                colors_ += ["#5e4fa2"]

        colors_diff = []

        for i in range(len(y)):   
            if(y_pred[i]==y[i]):        
                colors_diff += ["#006837"]
            else:
                colors_diff += ["#a50026"]    


print("\narr1 = ", arr1 )
print("\narr2 = ", arr2 )
print("\narr3 = ", arr3 )
print("\narr4 = ", arr4 )
print("\narr5 = ", arr5 )
print("\narr6 = ", arr6 )

c1_acc_mean = np.mean(acc_arr1)
c1_sns_mean = np.mean(arr_sns1)
c1_prec_mean = np.mean(arr_prec1)
c1_F1_mean = np.mean(arr_F1_1)
c1_auc_mean = np.mean(arr_auc1)
c1_train_mean = np.mean(arr_tran_t1) * 200
c1_test_mean = np.mean(arr_test_t1) * 200

print("\nc1_acc_mean = ", c1_acc_mean, "\n")
print("\nc1_acc_mean = ", c1_sns_mean, "\n")

plotdata = pd.DataFrame({"KNN":[c1_acc_mean, c1_sns_mean, c1_prec_mean, c1_F1_mean, c1_auc_mean, c1_train_mean, c1_test_mean]}, index=["acc", "rec", "prec", "f1", "roc_auc","train_t","test_t"])

plotdata.plot(kind="bar")
plt.show()


# roc : 
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
model = gs_results.best_estimator_
model.fit(X_train, y_train)
    
probs = model.predict_proba(X_test)
   
probs = probs[:, 1]
    
auc = roc_auc_score(y_test, probs)
    
print('AUC: %.3f' % auc)   
    
fpr, tpr, thresholds = roc_curve(y_test, probs)    
    
plt.plot([0, 1], [0, 1], linestyle='--')   
    
lines = plt.plot(fpr, tpr, marker='.')    
    
plt.title(gs_results.best_estimator_)    
    
plt.xlabel('False Positive Rate')
    
plt.ylabel('True Positive Rate')
    
plt.legend(lines[:2], ['AUC = %.3f' % auc, 'second']);
    
plt.show()

# krzywa dyskryminacyjna : 
    
h = .02  
    
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  
    
Z = Z.reshape(xx.shape)    
    
plt.figure()   
    
plt.pcolormesh(xx, yy, Z, shading='auto', cmap=cmap_light)    
     
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    
plt.xlim(xx.min(), xx.max())
    
plt.ylim(yy.min(), yy.max())
    
plt.show()