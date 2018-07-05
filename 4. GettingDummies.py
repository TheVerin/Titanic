#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Getting dummies for title
def title_dummies_X():
    global X
    title_dummies = pd.get_dummies(X['Title'], prefix = 'Title')
    X = pd.concat([X, title_dummies], axis = 1)
    X.drop('Title', axis = 1, inplace = True)
    return X
X = title_dummies_X()

#Embarked
def process_embarked_X():
    global X
    X.Embarked.fillna('S', inplace = True)
    embarked_dummies = pd.get_dummies(X['Embarked'], prefix = 'Embarked')
    X = pd.concat([X, embarked_dummies], axis = 1)
    X.drop(['Embarked'], axis = 1, inplace = True)
    return X
X = process_embarked_X()

#Dummies for class
def process_class_X():
    global X
    pclass_dummies = pd.get_dummies(X['Pclass'], prefix = 'Pclass')
    X = pd.concat([X, pclass_dummies], axis = 1)
    return X
X = process_class_X()
X.drop(['Pclass'], axis = 1, inplace = True)
