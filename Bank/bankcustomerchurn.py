# -*- coding: utf-8 -*-
# Bank Customer Churn Analysis

# 1. Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 2. Loading the Dataset
dataset = pd.read_csv('/content/Churn_Modelling.csv')

# 3. Exploring the Dataset
print(dataset.head())
print(dataset.tail())
print(dataset.shape)
print(dataset.size)
print(dataset.dtypes)
print(dataset.info())
print(dataset.describe())
print(dataset.columns)
print(dataset.isnull().sum())
print(dataset.duplicated().sum())
print(dataset.nunique())

# 4. Analyzing Categorical Variables
print(dataset.Geography.unique())
print(dataset.Gender.unique())
print(dataset.Exited.unique())
print(dataset.IsActiveMember.unique())
print(dataset.HasCrCard.unique())
print(dataset.NumOfProducts.unique())
print(dataset.Tenure.value_counts().sort_index())
print(dataset.CreditScore.value_counts().sort_index())
print(dataset.IsActiveMember.value_counts())
print(dataset.HasCrCard.value_counts())

# 5. Data Visualization
fig = px.histogram(dataset, x='Geography', title='Geographic Distribution')
fig.show()

age = px.histogram(dataset, x='Age', title='Age Distribution')
age.update_layout(bargap=0.5)
age.show()

crd = px.histogram(dataset, x='CreditScore', title='Credit Score Distribution')
crd.update_layout(bargap=0.5)
crd.show()

product_counts = dataset['NumOfProducts'].value_counts().reset_index()
product_counts.columns = ['NumOfProducts', 'Count']
pdt = px.pie(product_counts, values='Count', names='NumOfProducts', title='Distribution of Number of Products')
pdt.show()

sns.countplot(x='Gender', data=dataset, palette='Blues_d')
plt.show()

sns.countplot(x='HasCrCard', data=dataset, palette='Blues_d')
plt.show()

sns.distplot(dataset['Tenure'])
plt.show()

sns.countplot(x='IsActiveMember', data=dataset, palette='Blues_d')
plt.show()

sns.distplot(dataset['Balance'])
plt.show()

# 6. Data Cleaning
dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 7. Encoding Categorical Features
encoder = LabelEncoder()
dataset['Geography'] = encoder.fit_transform(dataset['Geography'])
dataset['Gender'] = encoder.fit_transform(dataset['Gender'])

# 8. Preparing Features and Labels
X = dataset.drop('Exited', axis=1)
Y = dataset['Exited']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

# 9. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 10. Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
log_predict = log_reg.predict(X_test)

# 11. Evaluating Logistic Regression
print(f"Accuracy: {accuracy_score(Y_test, log_predict)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, log_predict)}")
print(f"Classification Report:\n{classification_report(Y_test, log_predict)}")
print(f"Cross-validation (5-fold): {cross_val_score(log_reg, X, Y, cv=5, scoring='accuracy').mean()}")

# 12. Decision Tree Model
des_tree = DecisionTreeClassifier()
des_tree.fit(X_train, Y_train)
des_predict = des_tree.predict(X)

# 13. Evaluating Decision Tree
print(f"Accuracy: {accuracy_score(Y, des_predict)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y, des_predict)}")
print(f"Classification Report:\n{classification_report(Y, des_predict)}")
print(f"Cross-validation (5-fold): {cross_val_score(des_tree, X, Y, cv=5, scoring='accuracy').mean()}")

# 14. Random Forest Model
ran_cls = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
ran_cls.fit(X_train, Y_train)
ran_predict = ran_cls.predict(X_test)

# 15. Evaluating Random Forest
print(f"Accuracy: {accuracy_score(Y_test, ran_predict)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, ran_predict)}")
print(f"Classification Report:\n{classification_report(Y_test, ran_predict)}")
print(f"Cross-validation (5-fold): {cross_val_score(ran_cls, X, Y, cv=5, scoring='accuracy').mean()}")

# 16. Gradient Boosting Model
gnd_bst = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)
gnd_bst.fit(X_train, Y_train)
gnd_predict = gnd_bst.predict(X_test)

# 17. Evaluating Gradient Boosting
print(f"Accuracy: {accuracy_score(Y_test, gnd_predict)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, gnd_predict)}")
print(f"Classification Report:\n{classification_report(Y_test, gnd_predict)}")
print(f"Cross-validation (5-fold): {cross_val_score(gnd_bst, X, Y, cv=5, scoring='accuracy').mean()}")
