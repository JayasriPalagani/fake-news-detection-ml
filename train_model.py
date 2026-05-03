import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
fake = pd.read_csv(r"D:\fakenewsdetect\fake.csv")
true = pd.read_csv(r"D:\fakenewsdetect\true.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1)

X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")