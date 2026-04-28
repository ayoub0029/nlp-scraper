import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv("data/bbc_news_train.csv")

X = train_df["Text"]
y = train_df["Category"]

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print("Train:", model.score(X_train, y_train))
print("Test:", model.score(X_test, y_test))

train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(max_iter=1000),
    X_vec,
    y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label="Train Accuracy")
plt.plot(train_sizes, test_mean, label="Validation Accuracy")

plt.title("Learning Curves - Topic Classifier")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("results/learning_curves.png")
plt.close()

pickle.dump(model, open("results/topic_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("results/vectorizer.pkl", "wb"))


if __name__ == "__main__":
    print("Training complete")