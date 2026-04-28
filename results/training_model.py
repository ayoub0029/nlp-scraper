import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("data/bbc_news_train.csv")

X = train_df["Text"]
y = train_df["Category"]

# convert text to numbers
vectorizer = CountVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

pickle.dump(model, open("results/topic_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("results/vectorizer.pkl", "wb"))