from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import time

#import dataa
df = pd.read_csv('/Users/wayne/Downloads/winequality-white.csv', sep=';')
#Convert dataframe to matrix.
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].as_matrix()

#cd_metric = cosine_distances(X, X)
#Conver series to array.
y = df['quality'].values

#Start resubstitution
#Create six different models.
brute_manhattan = KNeighborsClassifier(algorithm='brute', metric='manhattan')
brute_euclidean = KNeighborsClassifier(algorithm='brute', metric='euclidean')
#brute_cosine = KNeighborsClassifier(algorithm='brute', metric=)
kd_manhattan = KNeighborsClassifier(algorithm='kd_tree', metric='manhattan')
kd_euclidean = KNeighborsClassifier(algorithm='kd_tree', metric='euclidean')
#kd_cosine = KNeighborsClassifier(algorithm='kd_tree', metric=cd_metric)

classfiers = {"Brute_Manhattan": brute_manhattan, "Brute_Euclidean": brute_euclidean,
              "KD_tree_Manhattan": kd_manhattan, "KD_tree_euclidean": kd_euclidean}

for key, classfier in classfiers.items():
    print('Resubstitution result for ' + key + ' K-NN classfier')
    myClassfier = classfier
    start = time.time()
    myClassfier.fit(X, y)
    totalTime = time.time() - start
    print('Time for training and querying: ' + str(totalTime))
    predicted = myClassfier.predict(X)
    myConfusionMatrix = confusion_matrix(y, predicted)
    print('Confusion Matrix:')
    print(myConfusionMatrix)

#Start K-fold
kf = KFold(n_splits=10, shuffle=True)
for key, classfier in classfiers.items():
    myConfusionMatrix = np.array([[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]])

    print('K-fold result for ' + key + ' K-NN classfier')
    start = time.time()
    for train_index, test_index in kf.split(X):
        myClassfier = classfier
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        myClassfier.fit(X_train, y_train)
        predicted = myClassfier.predict(X_test)
        temp = confusion_matrix(y_test, predicted, labels=[6, 5, 7, 8, 4, 3, 9])
        myConfusionMatrix = myConfusionMatrix + temp
    totalTime = time.time() - start
    print('Time for training and querying: ' + str(totalTime))
    print('Confusion Matrix:')
    print(myConfusionMatrix)
