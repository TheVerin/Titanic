#importing libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

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
