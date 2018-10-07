#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:37:22 2018

@author: chinmayikm
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import time
import numpy as np
from sklearn.model_selection import KFold

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


inSampleScore=[]
outSampleScore=[]

#Decision Tree Classifier

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    y_test_pred = tree.predict(X_test)
    y_train_pred = tree.predict(X_train)    
    In_score = metrics.accuracy_score(y_train, y_train_pred)
    Out_score = metrics.accuracy_score(y_test, y_test_pred)
    inSampleScore.append(In_score)    
    inSampleScore.append(Out_score)
    print('Random State :: %d, In-sample score:: %.3f, Out-of-sample score:: %.3f' % (i,In_score,Out_score))

print("Mean In-Sample :: ",np.mean(inSampleScore))
print("Standard Deviation In-Sample ::",np.std(inSampleScore))
print("Mean Out-of-Sample ::",np.mean(inSampleScore))
print("Standard Deviation Out-of-sample ::",np.std(inSampleScore))



#K Fold cross validation
for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=0.1,
                                                    random_state=i,
                                                    stratify=y)


    kfold = KFold(n_splits=10,random_state=i).split(X_train, y_train)

    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

    tree.fit(X_train, y_train)

    Scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=10, n_jobs=1)

    y_test_pred = tree.predict(X_test)
    Out_of_Sample_Score=metrics.accuracy_score(y_test,y_test_pred)

    print('Random State :%d',i,'\n')
    print('CV accuracy scores: %s' % Scores,'\n')
    
    print("Out-of-Sample Score:", Out_of_Sample_Score,'\n')


print('CV accuracy mean/std: %.3f +/- %.3f' % (np.mean(Scores), np.std(Scores)),'\n')


print("My name is Chinmayi Kargal Manjunath")
print("My NetID is: ck21")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")