import numpy as np
from sklearn import datasets, linear_model, tree
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score

iris = datasets.load_iris()

X = iris.data
Y = iris.target
number_of_samples = len(Y)
#splitting into training and test sets
random_indice = np.random.permutation(number_of_samples)
#Traing set
num_training_samples = int(number_of_samples * 0.6)
x_train = X[random_indice[:num_training_samples]]
y_train = Y[random_indice[:num_training_samples]]
#Test set
num_test_samples = int(number_of_samples * 0.4)
x_test = X[random_indice[-num_test_samples:]]
y_test = Y[random_indice[-num_test_samples:]]

#Create the model and training
model_resub = tree.DecisionTreeClassifier()
model_resub.fit(x_train, y_train)

print ("Result for resubstitution")
#Get all resubstitution validation result
predicted = model_resub.predict(x_test)
print("Predict result :")
print(predicted)

confusion_matrix_resub = metrics.confusion_matrix(y_test, predicted)
print("The confusion_matrix for only split traing/testing is :")
print(confusion_matrix_resub)

print ("============================================")

print("Result for k-fold-cross validation :")
#Get all k-fold-cross validation result
kf = KFold(n_splits=10)

n = 1
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    model_kfold = tree.DecisionTreeClassifier()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model_kfold.fit(X_train, y_train)
    predicted = model_kfold.predict(X_test)
    print ("Round ", n, " Result :")
    print("Predict result :")
    print(predicted)
    confusion_matrix_kfold = metrics.confusion_matrix(y_test, predicted)
    print("The confusion_matrix for k-fold round ", n, "is :")
    print(confusion_matrix_kfold)
    n = n + 1
