#importing libraries
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

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