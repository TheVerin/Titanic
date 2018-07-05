#importing libraries
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

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