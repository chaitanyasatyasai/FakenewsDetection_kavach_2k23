# Fake News Detection Project
This project focuses on building a machine learning model to classify news articles as Fake or Real using natural language processing (NLP) and machine learning techniques.

# Table of Contents
Introduction
Dataset
Requirements
Preprocessing
Modeling
Evaluation
Usage
Results
Team Members

# Introduction
Fake news has become a significant issue in today's digital age. This project aims to create a model that can accurately classify news articles as Fake or Real based on their content using NLP techniques and a Multinomial Naive Bayes classifier.

# Dataset
The project uses two datasets:

Fake.csv: Contains articles labeled as fake.
True.csv: Contains articles labeled as real.
The datasets include columns like text, subject, and date, which are used for training the model.

# Requirements
The project requires the following libraries:

Python (3.x)
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
Keras
NLTK
Spacy
Tqdm
# Google Colab (for mounting Google Drive and accessing datasets)
Install the required libraries using:

pip install numpy pandas matplotlib seaborn scikit-learn keras nltk spacy tqdm

Download the English language model for Spacy:

python -m spacy download en_core_web_sm

# Preprocessing
Data Cleaning: Handle missing values and remove duplicates.
Text Preprocessing: Tokenization, lemmatization, and stop word removal using Spacy.
Label Encoding: Fake news is labeled as 1 and real news as 0.
# Modeling
  # Vectorization:
    CountVectorizer: Transforms text into a bag-of-words model.
    TfidfVectorizer: Converts text into TF-IDF feature vectors for improved performance.
  # Classifier:
     Multinomial Naive Bayes: Used for training and classifying the text data.
# Evaluation
The model is evaluated using:

Accuracy Score: Measures the proportion of correctly classified articles.
Confusion Matrix: Visualizes the performance of the model.
Classification Report: Provides precision, recall, and F1-score for each class.
# Usage
To use this project:

Mount Google Drive and load the datasets:

from google.colab import drive
drive.mount('/content/drive')

Load and preprocess the datasets using the provided code.
Train the model with the preprocessed data.
Evaluate the model using the test set.

Predict the label of a new news article:

article = input("Please enter a news article:")

# Results
The model achieves an accuracy of approximately (95.6%) using the TF-IDF vectorizer and Multinomial Naive Bayes classifier.

# Team Members
sashank
kalyan
vamsi Krishna
sravani
sripriya
chaitanya
