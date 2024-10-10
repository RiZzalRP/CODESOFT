# -*- coding: utf-8 -*-
# Movie Genre Classification

# 1. Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 2. Loading the Dataset
title = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
train_data = pd.read_csv("//content/train_data.txt", sep=":::", engine="python", names=title)

title = ['ID', 'TITLE', 'DESCRIPTION']
test_data = pd.read_csv("/content/test_data.txt", sep=":::", engine="python", names=title)

# 3. Initial Data Exploration
print(train_data.head())
print(test_data.head())

# 4. Data Types and Shapes
print(train_data.dtypes)
print(test_data.dtypes)
print(train_data.shape)
print(test_data.shape)

# 5. Data Information Summary
print(train_data.info(), '\n')
print(test_data.info())

# 6. Descriptive Statistics
print("Train Data \n", train_data.describe(), '\n')
print("Test Data \n", test_data.describe())

# 7. Counting Non-null Entries
print(train_data.count())
print(test_data.count())

# 8. Checking for Missing Values
print(train_data.isnull().sum())
print(test_data.isnull().sum())

# 9. Checking for Duplicates
print(train_data.duplicated().sum())
print(test_data.duplicated().sum())

# 10. Grouping Data by Genre
print(train_data.groupby('GENRE').describe())

# 11. Data Visualization: Genre Distribution
genre_counts = train_data['GENRE'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']
fig = px.bar(genre_counts, x='Genre', y='Count', title='Most Watched Genres',
             labels={'Genre': 'Genre', 'Count': 'Number of Movies'},
             color_discrete_sequence=['lightgreen'])
fig.show()

# 12. Visualization: Top 10 Most Watched Genres
top_genres = train_data['GENRE'].value_counts().nlargest(10).reset_index()
top_genres.columns = ['Genre', 'Count']
plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Genre', data=top_genres, palette='Greens')
plt.title('Top 10 Most Watched Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

# 13. Visualization: Top 10 Unwatched Genres
top_genres = train_data['GENRE'].value_counts().nsmallest(10).reset_index()
top_genres.columns = ['Genre', 'Count']
plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Genre', data=top_genres, palette='Blues')
plt.title('Top 10 Unwatched Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

# 14. Encoding Genre Labels
encoder = LabelEncoder()
train_data['GENRE'] = encoder.fit_transform(train_data['GENRE'])

# 15. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_Tfidf = tfidf.fit_transform(train_data['DESCRIPTION']).toarray()
X_test_Tfidf = tfidf.transform(test_data['DESCRIPTION']).toarray()

# 16. Preparing Features and Labels
X = X_train_Tfidf
Y = train_data['GENRE']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42, shuffle=True)

# 17. Logistic Regression Model Training
Log_Reg = LogisticRegression(solver='liblinear', multi_class='ovr')
Log_Reg.fit(X_train, Y_train)
Log_Predct = Log_Reg.predict(X_test)

# 18. Evaluating Logistic Regression Model
print(f"Accuracy: {accuracy_score(Y_test, Log_Predct)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Log_Predct)}")
print(f"Classification Report:\n{classification_report(Y_test, Log_Predct)}")
print(f"Cross-validation (5-fold): {cross_val_score(Log_Reg, X, Y, cv=5, scoring='accuracy').mean()}")

# 19. Naive Bayes Model Training
Nav_Bys = MultinomialNB()
Nav_Bys.fit(X_train, Y_train)
Nav_Predct = Nav_Bys.predict(X_test)

# 20. Evaluating Naive Bayes Model
print(f"Accuracy: {accuracy_score(Y_test, Nav_Predct)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Nav_Predct)}")
print(f"Classification Report:\n{classification_report(Y_test, Nav_Predct)}")
print(f"Cross-validation (5-fold): {cross_val_score(Nav_Bys, X, Y, cv=5, scoring='accuracy').mean()}")

# 21. Support Vector Classifier Model Training
SVC_Model = SVC(kernel='linear', gamma='auto')
SVC_Model.fit(X_train, Y_train)
SVC_Predct = SVC_Model.predict(X_test)

# 22. Evaluating Support Vector Classifier Model
print(f"Accuracy: {accuracy_score(Y_test, SVC_Predct)}")
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, SVC_Predct)}")
print(f"Classification Report:\n{classification_report(Y_test, SVC_Predct)}")
print(f"Cross-validation (5-fold): {cross_val_score(SVC_Model, X, Y, cv=5, scoring='accuracy').mean()}")

# 23. Predicting Test Data Genres with Logistic Regression
Text_Predction = Log_Reg.predict(X_test_Tfidf)
print(Text_Predction)
