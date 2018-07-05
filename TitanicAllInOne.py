#Titanic

#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Getting dataset
Titanic = pd.read_csv('train.csv')

#Visualiations:
#Percentage of survivers in each gender
Titanic['Died'] = 1 -Titanic['Survived']
Titanic.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind = 'bar', stacked = True, colors = ['g', 'r'])

#Violinplot containing corelation between gender and age in case of surviving
sns.violinplot(x = 'Sex', y = 'Age', hue = 'Survived', data = Titanic, split = True, palette = {0:'r', 1:'g'})

#Corelation between cost of ticket and ability to survive
plt.hist([Titanic[Titanic['Survived'] == 1]['Fare'], Titanic[Titanic['Survived'] == 0]['Fare']], 
         stacked = True, color = ['g', 'r'], bins = 50, label = ['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Numbers of passangers')
plt.legend()

#Comparing Age and cost of ticket with ability to survive
myPlot = plt.subplot()
myPlot.scatter(Titanic[Titanic['Survived'] == 1]['Age'], Titanic[Titanic['Survived'] == 1]['Fare'], c = 'g')
myPlot.scatter(Titanic[Titanic['Survived'] == 0]['Age'], Titanic[Titanic['Survived'] == 0]['Fare'], c = 'r')

#Corelation between cost of a ticket and class
myPlot = plt.subplot()
myPlot.set_label('Average fare')
Titanic.groupby('Pclass').mean()['Fare'].plot(kind = 'bar', ax = myPlot)

#Corelation between embarked and cost corelates with surviving
sns.violinplot(x = 'Embarked', y = 'Fare', hue = 'Survived', data = Titanic, split = True, palette = {0:'r', 1:'g'})

#Thanks that, I know that the most important values are: age, cost of a ticket and gender. Embarktation is worth less than them. :D

#Importing the right data    
#Making a method wchih allows us to import data

def combined_data():
    train = pd.read_csv('train.csv')
    train.drop('Survived', axis = 1, inplace = True)
    
    test = pd.read_csv('test.csv')
    X = train.append(test)
    X.reset_index(inplace = True)
    X.drop(['index', 'PassengerId'], axis = 1, inplace = True)

    return X
X = combined_data()

Y = Titanic.iloc[:, 1].values

#Changind gender into 0,1 by feature
def process_sex():
    global X
    # mapping string values to numerical one 
    X['Sex'] = X['Sex'].map({'male':1, 'female':0})
    return X
X = process_sex()

#Let's look into Name column -> each person has a title 0.0, and it is the second string in text

titles = set()
for name in Titanic['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)
#Now let's group this titles -> Crew, Royality, Mrs, Miss, Master, Mr, Ms
Title_Dictionary = {'Rev':'Crew', 
                    'Miss':'Miss',
                    'Dr':'Crew',
                    'Master':'Kid',
                    'Don':'Royality',
                    'Major':'Crew',
                    'Col':'Crew',
                    'Capt':'Crew',
                    'the Countess':'Royality',
                    'Mme':'Mrs',
                    'Lady':'Royality',
                    'Mr':'Mr',
                    'Jonkheer':'Royality',
                    'Mlle':'Miss',
                    'Ms':'Mrs',
                    'Mrs':'Mrs',
                    'Sir':'Royality'}

#Making features to set titles column in each dataset
def get_titled_X():
    global X
    X['Title'] = X['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    X['Title'] = X.Title.map(Title_Dictionary)
    return X
X = get_titled_X()

#Checking is there any person without title
X[X['Title'].isnull()]
X.Title.fillna('Mrs', inplace = True)

#Adding a family into model
def process_family():
    global X
    X['FamilySize'] = X['Parch'] + X['SibSp'] + 1
    X['Single'] = X['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    X['SmallFamily'] = X['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    X['BigFamily'] = X['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    return X
X = process_family()
X.drop(['FamilySize'], axis = 1, inplace = True)


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

#Getting dummies for title
def title_dummies_X():
    global X
    title_dummies = pd.get_dummies(X['Title'], prefix = 'Title')
    X = pd.concat([X, title_dummies], axis = 1)
    X.drop('Title', axis = 1, inplace = True)
    return X
X = title_dummies_X()

#Fare
def process_fares_X():
    global X
    X.Fare.fillna(X.iloc[:].Fare.mean(), inplace = True)
    return X
X = process_fares_X()

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

#Removing columns
X.drop([ 'Cabin', 'Ticket', 'Name', 'Parch', 'SibSp'], inplace = True, axis = 1)

#Splitting for test and train set
Z = X.iloc[891:, :]
X = X.iloc[:891, :]


#Scalling variables
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X[:, 0:] = X_sc.fit_transform(X[:, 0:])


#Splitting dataset into training and validate set
from sklearn.cross_validation import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#Implementing logistic regression model
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0, C = 10, max_iter = 200)
classifier_LR.fit(X_train, Y_train)
#Predicting the test set results
Y_pred_LR = classifier_LR.predict(X_val)


#Implementing a gradient boosting classification
from xgboost import XGBClassifier
classifier_GB = XGBClassifier(max_depth = 10, n_estimators = 300, learning_rate = 0.1, base_score = 0.6, min_child_weight = 5)
classifier_GB.fit(X_train, Y_train)
#Predicting the test set results
Y_pred_GB = classifier_GB.predict(X_val)


#Implementing Random Forest Method
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 500, criterion = 'gini', random_state = 0, class_weight = 'balanced', verbose = 1)
classifier_RF.fit(X_train, Y_train)
#Predicting the test set results
Y_pred_RF = classifier_RF.predict(X_val)


#Report of the prediction -> Logistic Regression = LR, Gradient Boosting = GB, Random Forest = RF
#Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve
cm_LR = confusion_matrix(y_true = Y_val, y_pred = Y_pred_LR)
cm_GB = confusion_matrix(y_true = Y_val, y_pred = Y_pred_GB)
cm_RF = confusion_matrix(y_true = Y_val, y_pred = Y_pred_RF)
#Classification report
report_LR = classification_report(Y_val, Y_pred_LR)
report_GB = classification_report(Y_val, Y_pred_GB)
report_RF = classification_report(Y_val, Y_pred_RF)
#Accuracy score
accuracy_score(Y_val, Y_pred_LR)
accuracy_score(Y_val, Y_pred_GB)
accuracy_score(Y_val, Y_pred_RF)
#Visualising results by ROC curve
x1_LR, x2_LR, _ = roc_curve(Y_val, Y_pred_LR)
x1_GB, x2_GB, _ = roc_curve(Y_val, Y_pred_GB)
x1_RF, x2_RF, _ = roc_curve(Y_val, Y_pred_RF)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x1_LR, x2_LR, label='LR')
plt.plot(x1_GB, x2_GB, label='GB')
plt.plot(x1_RF, x2_RF, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



#Implementing logistic regression model
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0, C = 10, max_iter = 200)
classifier_LR.fit(X, Y)


#Implementing a gradient boosting classification
from xgboost import XGBClassifier
classifier_GB = XGBClassifier(max_depth = 10, n_estimators = 300, learning_rate = 0.1, base_score = 0.6, min_child_weight = 5)
classifier_GB.fit(X, Y)


#Implementing Random Forest Method
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 500, criterion = 'gini', random_state = 0, class_weight = 'balanced', verbose = 1)
classifier_RF.fit(X, Y)


LR = classifier_LR.predict(Z)
df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = LR
df_output[['PassengerId','Survived']].to_csv('LogReg.csv', index=False)

GB = classifier_GB.predict(Z)
df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = GB
df_output[['PassengerId','Survived']].to_csv('XGB.csv', index=False)

RF = classifier_RF.predict(Z)
df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = RF
df_output[['PassengerId','Survived']].to_csv('RanFor.csv', index=False)

