
import pandas as pd
import string
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load and clean dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["label", "text"]]
df.dropna(inplace=True)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Inject modern spam examples
extra_spam = [
    "You’ve won a $1000 Walmart gift card! Click here to claim.",
    "Your account has been suspended. Click to verify now.",
    "Limited offer! Visit www.getmoneyfast.biz to win cash.",
    "Congratulations! You've been selected to receive a free iPhone!",
    "Are you unique enough? Find out now at www.areyouunique.co.uk",
    "Your electricity bill is due. Pay immediately to avoid disconnection.",
]
extra_df = pd.DataFrame({'label': [1]*len(extra_spam), 'text': extra_spam})
df = pd.concat([df, extra_df], ignore_index=True)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df["cleaned_text"] = df["text"].apply(preprocess_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = tfidf.fit_transform(df["cleaned_text"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train MultinomialNB
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train, y_train)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate MultinomialNB
nb_pred = nb_model.predict(X_test)
print("=== MultinomialNB Classification Report ===")
print(classification_report(y_test, nb_pred, target_names=["HAM", "SPAM"]))
print(f"MultinomialNB Accuracy: {accuracy_score(y_test, nb_pred):.4f}")

# Evaluate Logistic Regression
lr_pred = lr_model.predict(X_test)
print("\n=== Logistic Regression Classification Report ===")
print(classification_report(y_test, lr_pred, target_names=["HAM", "SPAM"]))
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")

# Save best model and vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
if accuracy_score(y_test, lr_pred) > accuracy_score(y_test, nb_pred):
    joblib.dump(lr_model, "spam_model.pkl")
else:
    joblib.dump(nb_model, "spam_model.pkl")
print("Model and vectorizer saved.")
