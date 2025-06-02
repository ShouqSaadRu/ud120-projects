#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))

from email_preprocess import preprocess

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here

model = GaussianNB()

t0 = time()
model.fit(features_train,labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
actual_pred = model.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(labels_test , actual_pred )
print("accuracy: ", accuracy)


