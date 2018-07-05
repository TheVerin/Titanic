#importing libraries
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
