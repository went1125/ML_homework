import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

#Part A
## Data preprocessing
X_train = np.array(
[[168, 3, 1814, 15, 0.001, 1879],
 [156, 3, 1358, 14, 0.01, 1425],
 [176, 3.5, 2200, 16, 0.005, 2140],
 [256, 3, 2070, 27, 0.2, 2700],
 [230, 5, 1410, 131, 3.5, 1575],
 [116, 3, 1238, 104, 0.06, 1221],
 [242, 7, 1315, 104, 0.01, 1434],
 [242, 4.5, 1183, 78, 0.02, 1374],
 [174, 2.5, 1110, 73, 1.5, 1256],
 [1004, 35, 1218, 81, 1172, 33.3],
 [1228, 46, 1889, 82.4, 1932, 43.1],
 [964, 17, 2120, 20, 1030, 1966],
 [2008, 32, 1257, 13, 1038, 1289]])

Binning_X = np.array(
[[0, 0, 2, 0, 0, 2],
 [0, 0, 1, 1, 0, 1],
 [0, 0, 2, 1, 0, 2],
 [0, 0, 2, 1, 0, 2],
 [0, 0, 1, 2, 0, 1],
 [0, 0, 1, 2, 0, 1],
 [0, 0, 1, 2, 0, 1],
 [0, 0, 1, 1, 0, 1],
 [0, 0, 1, 1, 0, 1],
 [1, 2, 1, 1, 1, 0],
 [1, 2, 2, 1, 2, 0],
 [1, 1, 2, 0, 1, 2],
 [2, 2, 1, 0, 1, 1]])
y_train = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3])

X_test = np.array([222, 4.5, 1518, 74, 0.25, 1642])
Binning_X_test = np.array([0, 0, 2, 1, 0, 1])

print("PartA: Using Normal distribution")
clf = GaussianNB()
clf.fit(X_train, y_train)
target_pred = clf.predict([X_test])
print(target_pred)

print("PartA: Binning with Laplace smoothing:")
clf = MultinomialNB() ##Using Laplace smoothing method.
clf.fit(Binning_X, y_train)
target_pred = clf.predict([Binning_X_test])
print(target_pred)
#Part A End


#Part B
col_name = ['Class', 'Q-E', 'ZN-E', 'PH-E', 'DBO-E','DQO-E', 'SS-E', 'SSV-E', 'SED-E','COND-E', 'PH-P', 'DBO-P', 'SS-P',
            'SSV-P', 'SED-P', 'COND-P', 'PH-D','DBO-D', 'DQO-D', 'SS-D', 'SSV-D','SED-D', 'COND-D', 'PH-S', 'DBO-S',
            'DQO-S', 'SS-S', 'SSV-S', 'SED-S','COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P','RD-DBO-S', 'RD-DQO-S ',
            'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G']
## import data and split them
df = pd.read_csv('/Users/wayne/Downloads/test.csv', header=None)
df = df.replace('?',np.nan)
df.columns = col_name
df = df[col_name].apply(pd.to_numeric)
trainDataSet = df.loc[:503]
testDataSet = df.loc[504:]
trainDataSet.is_copy = False
testDataSet.is_copy = False

##Start data preprocessing
for value in col_name[1:]:
    ## fillna to mean value
    trainDataSet.loc[:,value] = trainDataSet[value].fillna(trainDataSet[value].mean())
    testDataSet.loc[:,value] = testDataSet[value].fillna(testDataSet[value].mean())

##Convert to ndarray
X_train = trainDataSet[col_name[1:]].as_matrix()
y_train = trainDataSet[col_name[0]].values

X_test = testDataSet[col_name[1:]].as_matrix()

print("PartB: Using Normal distribution")
clf = GaussianNB()
clf.fit(X_train, y_train)
target_pred = clf.predict(X_test)
print(target_pred)

## Data smoothing and binning
for value in col_name[1:]:
    ## fillna to mean value
    trainDataSet.loc[:,value] = pd.cut(trainDataSet[value], 6, labels=False)
    testDataSet.loc[:,value] = pd.cut(testDataSet[value], 6, labels=False)

X_train = trainDataSet[col_name[1:]].as_matrix()
X_test = testDataSet[col_name[1:]].as_matrix()

print("PartB: Binning with Laplace smoothing:")
clf = MultinomialNB() ##Using Laplace smoothing method.
clf.fit(X_train, y_train)
target_pred = clf.predict(X_test)
print(target_pred)
#Part B End
