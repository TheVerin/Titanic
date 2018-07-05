#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Dealing with missing values
#Age
grouped_train = X.iloc[:891].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

#Let's fill the NaN values with rigrt means
def fill_age(row):
    condition = (
            (grouped_median_train['Sex'] == row['Sex']) &
            (grouped_median_train['Title'] == row['Title']) &
            (grouped_median_train['Pclass'] == row['Pclass']))
    return grouped_median_train[condition]['Age'].values[0]

def process_age_X():
    global X
    X['Age'] = X.apply(lambda row: fill_age(row) 
    if np.isnan(row['Age']) else row['Age'], axis = 1)
    return X
X = process_age_X()

#Fare
def process_fares_X():
    global X
    X.Fare.fillna(X.iloc[:].Fare.mean(), inplace = True)
    return X
X = process_fares_X()