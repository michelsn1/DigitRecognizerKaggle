# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:05:20 2018

@author: miche
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')

#Criando os vetores de entrada e saída
X = dataset.iloc[:, 1:785].values
y = dataset.iloc[:, 0].values

#Checando se existe nan
dataset.isnull().values.any()

#Dividindo os dados em Training set e Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Ajustando o modelo de Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

classifier.score(X_test,y_test)


y_pred = classifier.predict(test)

submission=pd.DataFrame()
submission['ImageId']= test.index +1
submission['Label']=classifier.predict(test)
submission.to_csv('submission.csv',index=False)

