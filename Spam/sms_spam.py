# -*- coding: utf-8 -*-
# SMS Spam Detection Analysis

# 1. Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Loading the Dataset
dataset = pd.read_csv('/content/spam.csv', encoding='latin-1', engine='python', usecols=['v1', 'v2'])
dataset.columns = ['Label', 'Message']

# 3. Exploring the Dataset
print(dataset.head())
print(dataset.tail())
print(dataset.count())
print(dataset.shape)
print(dataset.size)
print(dataset.info())
print(dataset.describe())
print(dataset.dtypes)
print(dataset.isnull().sum())
print(dataset.Label.unique())
print(dataset.Label.value_counts())
print(dataset.duplicated().value_counts())
# 4. Removing Duplicated Rows
dataset = dataset.drop_duplicates(inplace=True)

# 5. Data Distribution Visualization
sns.displot(dataset['Label'], kde=True, color='orange')
sns.countplot(dataset['Label'], color='orange')

label = dataset.Label.value_counts()
plt.pie(label.values, labels=label.index, autopct='%1.2f%%')
plt.legend(loc='upper left', title='Weather', bbox_to_anchor=(1, 1))
plt.title("Label Graph")
plt.axis('equal')
plt.show()

# 6. Encoding Labels
encoder = LabelEncoder()
dataset['Label'] = encoder.fit_transform(dataset['Label'])

print(dataset.head())

# 7. Text Vectorization
vectorize = TfidfVectorizer(stop_words='english')
X = vectorize.fit_transform(dataset['Message']).toarray()
Y = dataset['Label']

# 8. Splitting the Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

# 9. Naive Bayes Classifier
Mul_NB = MultinomialNB()
Mul_NB.fit(X_train, Y_train)
Mul_pred = Mul_NB.predict(X_test)

# 10. Evaluating Naive Bayes
print(f"Training Score: {Mul_NB.score(X_train, Y_train)}")
print(f"Testing Score: {Mul_NB.score(X_test, Y_test)}")
print(f"Accuracy Score: {accuracy_score(Y_test, Mul_pred)}")
print(f"Confusion Matrix: {confusion_matrix(Y_test, Mul_pred)}")
print(f"Classification Report: {classification_report(Y_test, Mul_pred)}")
print(f'Cross-Validation: {cross_val_score(Mul_NB, X, Y, cv=5, scoring="accuracy").mean()}')

# 11. Logistic Regression Model
Log_Reg = LogisticRegression()
Log_Reg.fit(X_train, Y_train)
Log_pred = Log_Reg.predict(X_test)

# 12. Evaluating Logistic Regression
print(f"Training Score: {Log_Reg.score(X_train, Y_train)}")
print(f"Testing Score: {Log_Reg.score(X_test, Y_test)}")
print(f'Accuracy Score: {accuracy_score(Y_test, Log_pred)}')
print(f'Confusion Matrix: {confusion_matrix(Y_test, Log_pred)}')
print(f'Classification Report: {classification_report(Y_test, Log_pred)}')
print(f'Cross-Validation: {cross_val_score(Log_Reg, X, Y, cv=5, scoring="accuracy").mean()}')

# 13. SVM Model
SVM = SVC()
SVM.fit(X_train, Y_train)
SVM_pred = SVM.predict(X_test)

# 14. Evaluating SVM
print(f"Training Score: {SVM.score(X_train, Y_train)}")
print(f"Testing Score: {SVM.score(X_test, Y_test)}")
print(f'Accuracy Score: {accuracy_score(Y_test, SVM_pred)}')
print(f'Confusion Matrix: {confusion_matrix(Y_test, SVM_pred)}')
print(f'Classification Report: {classification_report(Y_test, SVM_pred)}')
print(f'Cross-Validation: {cross_val_score(SVM, X, Y, cv=5, scoring="accuracy").mean()}')
