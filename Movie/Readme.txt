🎬 Movie Genre Classification using Machine Learning 🎥📊

In this project, I applied machine learning techniques to classify movie genres based on their descriptions. The goal is to predict the genre of movies using natural language processing (NLP) and text data.

Project Overview:
Dataset: Movie titles, descriptions, and corresponding genres for training, and a test set with titles and descriptions for evaluation.
Objective: Predict the genre of a movie based on its description using classification algorithms.

Key Steps:

Data Exploration & Cleaning:

Loaded and explored train and test datasets.
Checked for null values, duplicates, and overall data structure.
Visualized the distribution of genres to understand the popularity of each genre.

Data Preprocessing:

Label Encoding: Converted genres into numerical labels for classification.
TF-IDF Vectorization: Used Term Frequency-Inverse Document Frequency (TF-IDF) to transform movie descriptions into numerical features for the models.

Splitting Data:

Split the data into training and testing sets with 80% training and 20% testing.

Model Training:

Logistic Regression: A multi-class model to handle multiple genres.
Naive Bayes: A fast and efficient model for text classification.
Support Vector Classifier (SVC): A non-linear classifier for better separation of classes.
Model Evaluation:

Evaluated models using accuracy score, confusion matrix, and classification report.
Implemented cross-validation to ensure stability and avoid overfitting.

Results:

Logistic Regression: Strong baseline performance with high accuracy and balanced precision-recall.
Naive Bayes: Efficient at classifying genres with textual data due to its simplicity.
SVC: Achieved competitive results, particularly with genres that have clear distinctions.

Next Steps:

Fine-tuning models to improve performance.
Exploring deep learning approaches like LSTM for genre prediction.
This project demonstrates how machine learning can be leveraged for text classification and has real-world applications in media categorization. 

