# -*- coding: utf-8 -*-
# Credit Card Fraud Detection Analysis

# 1. Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Loading Data
train_data = pd.read_csv('/content/fraudTrain.csv', header=0)
test_data = pd.read_csv('/content/fraudTest.csv', header=0)

# 3. Exploring Data
print(train_data.head())
print(test_data.head())
print(train_data.shape)
print(test_data.shape)
print(train_data.columns)
print(test_data.columns)
print(train_data.info())
print(test_data.info())
print(test_data.describe())
print(train_data.describe())
print(train_data.count())
print(test_data.count())
print(train_data.isnull().sum())
print(test_data.isnull().sum())
print(train_data.duplicated().sum())
print(test_data.duplicated().sum())
print(train_data.nunique())
print(train_data.dtypes)

# 4. Data Cleaning
train_data = train_data.dropna()
print(train_data.isnull().sum())

# 5. Combining Train and Test Datasets
dataset = pd.concat([train_data, test_data], ignore_index=True, axis=0)
print(dataset)
print(dataset.head())
print(dataset.tail())
print(dataset.shape)
print(dataset.size)
print(dataset.info())
print(dataset.describe())
print(dataset.duplicated().sum())

# 6. Dropping Unnecessary Columns
dataset = dataset.drop(['Unnamed: 0', 'trans_date_trans_time', 'job', 'first', 'last', 
                        'street', 'city', 'state', 'zip', 'dob', 'trans_num'], axis=1)
print(dataset.head())

# 7. Value Counts for Categorical Features
print(dataset['gender'].value_counts())
print(dataset['merchant'].value_counts())
print(dataset['category'].value_counts())

# 8. Data Visualization
sns.countplot(x='gender', data=dataset, palette='Greens')
plt.title('Gender Distribution')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='category', data=dataset, palette='Greens_d')
plt.title('Category Distribution')
plt.xticks(rotation=45, ha='right')
plt.show()

# 9. Correlation Heatmap for Numeric Data
numeric_data = dataset.select_dtypes(include=[np.number])
correlation = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Data')
plt.show()

# 10. Encoding Categorical Features
encoder = LabelEncoder()
dataset['gender'] = encoder.fit_transform(dataset['gender'])
dataset['category'] = encoder.fit_transform(dataset['category'])
dataset['merchant'] = encoder.fit_transform(dataset['merchant'])

# 11. Preparing Features and Labels
X = dataset.drop(['is_fraud'], axis=1)
Y = dataset['is_fraud']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

# 12. Logistic Regression Model
Log_Reg = LogisticRegression()
Log_Reg.fit(X_train, Y_train)
Log_Predct = Log_Reg.predict(X_test)

# 13. Evaluating Logistic Regression
print(f"Accuracy: {accuracy_score(Y_test, Log_Predct)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Log_Predct)}")
print(f"Classification Report:\n{classification_report(Y_test, Log_Predct)}")
print(f"Cross-validation (5-fold): {cross_val_score(Log_Reg, X, Y, cv=5, scoring='accuracy').mean()}")

# 14. Decision Tree Model
Des_Cls = DecisionTreeClassifier(random_state=42)
Des_Cls.fit(X_train, Y_train)
Des_Predct = Des_Cls.predict(X_test)

# 15. Evaluating Decision Tree
print(f"Accuracy: {accuracy_score(Y_test, Des_Predct)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Des_Predct)}")
print(f"Classification Report:\n{classification_report(Y_test, Des_Predct)}")
print(f"Cross-validation (5-fold): {cross_val_score(Des_Cls, X, Y, cv=5, scoring='accuracy').mean()}")

# 16. Random Forest Model
Ran_Cls = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
Ran_Cls.fit(X_train, Y_train)
Ran_Predct = Ran_Cls.predict(X_test)

# 17. Evaluating Random Forest
print(f"Accuracy: {accuracy_score(Y_test, Ran_Predct)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Ran_Predct)}")
print(f"Classification Report:\n{classification_report(Y_test, Ran_Predct)}")
print(f"Cross-validation (5-fold): {cross_val_score(Ran_Cls, X, Y, cv=5, scoring='accuracy').mean()}")
