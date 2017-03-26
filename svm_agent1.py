#import sys
#from class_vis import prettyPicture
#from prep_terrain_data import makeTerrainData


import random
import numpy as np
from sklearn import svm, preprocessing
import pandas as pd
from sklearn import metrics

from sklearn.model_selection import train_test_split

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier

#features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

def clean_outliers(df):
    print("Asizes:",df.shape)
    thedetail = df.detail0
    arange = range(1, thedetail.size+1)
    numbertodrop = []
    print("cleaning appoint with size:",arange)
    for ncount in arange:
        the_app = thedetail.loc[ncount]
        if (the_app < 0.2 or the_app > 100):    #('precision for agents:', 0.74/66/51 precision for buyers:', 0.502/43/48
            numbertodrop.append(ncount)
    print("drop these:",numbertodrop)
    df.drop(numbertodrop,axis = 0, inplace=True)
    print("Bsizes:",df.shape)
    return df

# def clean_list(df):
#     print("Asizes:",df.shape)
#     thelist = df.listPV0
#     arange = range(1, thelist.size+1)
#     numbertodrop = []
#     print("cleaning appoint with size:",arange)
#     for ncount in arange:
#         the_app = thelist.loc[ncount]
#         if the_app < 5:
#             numbertodrop.append(ncount)
#     print("drop these:",numbertodrop)
#     df.drop(numbertodrop,axis = 0, inplace=True)
#     print("Bsizes:",df.shape)
#     return df



#data = pd.DataFrame.from_csv("datatofit0320.csv")
data = pd.DataFrame.from_csv("datatofit0320nomalized.csv")   #LIST/20 DETAIL/10
#data.pop(tag)

#data = clean_outliers(data) #agents:', 0.81578947368421051)('precision for buyers:', 0.38528557599225555
#data = clean_list(data)
target = data.tag

data.drop('tag',axis=1, inplace=True)


print("data:",data)
print("sizes:",data.shape,target.shape)

random_state = random.randint(0,500)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=random_state)

print("sizes:",X_train.shape,y_train.shape,X_test.shape,y_test.shape)

clf = SVC(kernel="rbf" , class_weight= {1: 45}, degree=3,gamma="auto", C=0.63).fit(X_train, y_train) #63-87(72)/48-38(45) #6.7fits the correct number for 32 agents for rbf
#clf = SVC(kernel="rbf" , class_weight= {1: 100}).fit(X_train, y_train)   #A:68-85/B:43-37
rst = clf.predict(X_test)
count_agent = 0
for i, rst_agent in enumerate(rst):
    if rst_agent == 1:
        count_agent += 1
        np.set_printoptions(threshold='nan')
print("counted total:", count_agent)
print("YOUR SCORE ISSSSSS:",clf.score(X_test,y_test))
print("precision for agents:",metrics.precision_score(rst, y_test))
print("precision for buyers:",metrics.precision_score(rst, y_test, pos_label=0))
print("correct number:",metrics.accuracy_score(rst, y_test, normalize=False))
print("recall:",metrics.recall_score(rst, y_test))

nnt = GaussianNB().fit(X_train, y_train)   #92/30
rst_nnt = nnt.predict(X_test)
print("YOUR SCORE ISSSSSS:",nnt.score(X_test,y_test))
print("precision for agents:",metrics.precision_score(rst_nnt, y_test))
print("precision for buyers:",metrics.precision_score(rst_nnt, y_test, pos_label=0))
print("correct number:",metrics.accuracy_score(rst_nnt, y_test, normalize=False))
print("recall:",metrics.recall_score(rst_nnt, y_test))


# OUT PUT DATA TO ANALYSE
# X_test.to_csv('test0320.csv')
# y_test.to_csv('target0320.csv')
# for istr in rst:
#     print istr,","



# np.random.seed(0)
# X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
# y = [1] * 10 + [-1] * 10
# Z = np.random.randn(len(X))
# print("X:",X,"Z:",Z,"abs",abs(Z))
# print("y:",y)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data


#clf = svm.SVC()
# clf.fit(features_train, labels_train)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# rst = clf.predict(features_test)
# print rst
# #### store your predictions in a list named pred
# pred = rst
#
#
#
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)
#
# def submitAccuracy():
#     return acc