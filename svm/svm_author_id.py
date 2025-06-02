#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
from email_preprocess import preprocess

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

model = SVC(kernel= "rbf", C= 10000)

t0 = time()
model.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = model.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(labels_test , pred)

print("accuracy: " , accuracy)

