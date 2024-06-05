import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


dataset = pd.read_csv('./static/credit_rating.csv')

dataset.info()

dataset=dataset.drop(["S.No"], axis =1)
dataset=dataset.drop(["S.No."], axis =1)

dataset.head()

dataset.isnull().sum()

dataset.columns

"""# **Data Preprocessing**"""

#Converting the categorical data into numerical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['CHK_ACCT']=le.fit_transform(dataset['CHK_ACCT'])
dataset['History']=le.fit_transform(dataset['History'])
dataset['Purpose of credit']=le.fit_transform(dataset['Purpose of credit'])
dataset['Balance in Savings A/C']=le.fit_transform(dataset['Balance in Savings A/C'])
dataset['Employment']=le.fit_transform(dataset['Employment'])
dataset['Marital status']=le.fit_transform(dataset['Marital status'])
dataset['Co-applicant']=le.fit_transform(dataset['Co-applicant'])
dataset['Real Estate']=le.fit_transform(dataset['Real Estate'])
dataset['Marital status']=le.fit_transform(dataset['Marital status'])
dataset['Other installment']=le.fit_transform(dataset['Other installment'])
dataset['Residence']=le.fit_transform(dataset['Residence'])
dataset['Job']=le.fit_transform(dataset['Job'])
dataset['Phone']=le.fit_transform(dataset['Phone'])
dataset['Foreign']=le.fit_transform(dataset['Foreign'])
dataset['Credit classification']=le.fit_transform(dataset['Credit classification'])

dataset.describe()

dataset['Credit classification'].value_counts(normalize=True)*100

X = dataset.drop(['Credit classification'], axis = 1)
y = dataset['Credit classification']

print(X)

print(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train.shape,X_test.shape

"""# **Correlation Matrix**"""

#Feature Selection based on Correlation
X_train.corr()

#Check for multicollinearity(independent features should not be correlated that much based on threshold values)
plt.figure(figsize=(12,10))
corr=X_train.corr()
sns.heatmap(corr,annot=True)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test

"""# Performing HyperParameter Tuning"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification
from scipy.stats import randint

# Define the hyperparameter search space
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(10, 100),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': randint(1, 10),
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Create the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Create the random search object
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=100,  # Number of random combinations to try
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Evaluation metric
    n_jobs=-1,  # Use all available cores
    random_state=42,
    verbose=2
)

# Perform the random search
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_

# Get the best score
best_score = random_search.best_score_

print("Best hyperparameters:", best_params)
print("Best accuracy score:", best_score)

base_classifier = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=33, max_features=7, min_samples_leaf=5, min_samples_split=13, n_estimators=468)

# Initialize ensemble classifiers
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
boosting_classifier = AdaBoostClassifier(base_classifier, n_estimators=10, random_state=42)
voting_classifier = VotingClassifier(estimators=[('bagging', bagging_classifier), ('boosting', boosting_classifier)], voting='hard')

# Train the classifiers
bagging_classifier.fit(X_train, y_train)
boosting_classifier.fit(X_train, y_train)
voting_classifier.fit(X_train, y_train)

# Make predictions
bagging_predictions = bagging_classifier.predict(X_test)
boosting_predictions = boosting_classifier.predict(X_test)
voting_predictions = voting_classifier.predict(X_test)

# Calculate accuracies
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
boosting_accuracy = accuracy_score(y_test, boosting_predictions)
voting_accuracy = accuracy_score(y_test, voting_predictions)

# Print the accuracies
print("Bagging Accuracy:", bagging_accuracy)
print("Boosting Accuracy:", boosting_accuracy)
print("Voting Accuracy:", voting_accuracy)


#Printing Accuracy & f1_score
import seaborn as sns
accuracy = accuracy_score(y_test, boosting_predictions)
f1 = f1_score(y_test, boosting_predictions, average='weighted')
print("Model Accuracy: ", accuracy*100)
print("Model F1 score: ", f1*100)

target_names = dataset['Credit classification'].unique()
target_names

#Printing Confusion Matrix
print("Confusion Matrix")
conf_mat = confusion_matrix(y_test, boosting_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Creating Pickle file
import pickle
with open('RandomForestModel.pkl','wb') as f:
  pickle.dump(boosting_classifier, f)

import pickle
with open('scalar.pkl','wb') as f:
  pickle.dump(sc, f)

# features = dataset.columns.tolist()
# import pickle
# with open('features.pkl','wb') as f:
#   pickle.dump(features, f)


