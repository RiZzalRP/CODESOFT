 Exploring Genre Classification with Machine Learning! 🎬🎥

In my latest project, I developed machine learning models to predict movie genres using TF-IDF vectorization alongside Logistic Regression, Naive Bayes, and SVM algorithms.

Features:

ID: Unique movie identifier
TITLE: Movie title
GENRE: Target genre label (training set only)
DESCRIPTION: Textual descriptions for classification

Methods:
Data cleaning and addressing missing values
TF-IDF vectorization using 5,000 features
Label encoding for compatibility
Stratified train-test split (80/20)

Models Used:
Logistic Regression: Consistent performance with liblinear solver
Naive Bayes: Effective for text classification
SVC: Linear kernel for multi-class efficiency


Evaluation:
Accuracy, confusion matrix, and classification report metrics
5-fold cross-validation for reliability