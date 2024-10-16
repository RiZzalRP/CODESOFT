🚨 SMS Spam Detection using Machine Learning 📱🤖

In this project, I built models to detect spam messages using text data and several classification algorithms. This system can help in filtering unwanted messages and improving user experience.

Project Highlights:
Dataset: A collection of SMS messages labeled as "spam" or "ham" (non-spam).
Objective: Classify messages as either spam or not, using machine learning models.
Key Steps:
Data Exploration & Cleaning:

Checked for null values, duplicate rows, and the distribution of spam vs. non-spam messages.
Removed duplicates to ensure accurate results.
Data Preprocessing:

Label Encoding: Converted the target variable (spam/ham) into numeric format.
TF-IDF Vectorization: Transformed the text into numerical form using Term Frequency-Inverse Document Frequency (TF-IDF) for effective text classification.
Train-Test Split:

Split the dataset into 80% training and 20% testing data for reliable evaluation.
Models Built:

Naive Bayes: Performed well with text classification due to its simplicity and effectiveness.
Logistic Regression: A linear model that worked consistently across the dataset.
SVM (Support Vector Machine): A non-linear model that often excels in binary classification tasks.
Evaluation Metrics:

Accuracy Score: Measured how well each model classified spam messages.
Confusion Matrix: Evaluated the balance of false positives and false negatives.
Classification Reports: Assessed precision, recall, and F1-score.
Cross-Validation: Ensured model stability through 5-fold validation.
Results:
Naive Bayes demonstrated efficient spam detection due to its effectiveness in text-based classification.
SVM and Logistic Regression models also provided high accuracy and consistent performance across all evaluation metrics.
This project showcases the potential of machine learning to create systems that automatically filter spam, improving communication efficiency. 🚀

