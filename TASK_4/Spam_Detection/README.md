📧 Email Spam Detector using Naive Bayes

This is a simple yet effective machine learning project that classifies email text as SPAM or HAM (non-spam). The model uses Natural Language Processing (NLP) techniques along with a Multinomial Naive Bayes classifier.

🚀 Features

Preprocesses raw email text (lowercasing, punctuation removal, stopword removal, stemming).

Transforms text using TF-IDF vectorization.

Trains a spam classifier using Multinomial Naive Bayes.

Accepts multi-line input and predicts whether it's SPAM or HAM.

📁 Dataset

Make sure to have a dataset in CSV format with at least two columns:

text (email content)

label (spam or ham)

You can use the famous SMS Spam Collection Dataset from Kaggle.

Place your dataset in the project directory and name it spam.csv, or modify the path in the script.


