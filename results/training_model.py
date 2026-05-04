import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer


train_df = pd.read_csv("data/bbc_news_train.csv")


X = train_df["Text"]
y = train_df["Category"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        max_features=20000,
        sublinear_tf=True
    )),
    ("clf", LinearSVC(
        C=0.6,
        random_state=42
    ))
    # ("LR", LogisticRegression(max_iter=1000))
])

model_pipeline.fit(X_train, y_train)


y_pred = model_pipeline.predict(X_test)

print("Train:", model_pipeline.score(X_train, y_train))
print("Test:", model_pipeline.score(X_test, y_test))


test_df = pd.read_csv("data/bbc_news_tests.csv")

X_test_real = test_df["Text"]
y_test_real = test_df["Category"]

y_pred_real = model_pipeline.predict(X_test_real)

print("REAL TEST ACC:", accuracy_score(y_test_real, y_pred_real))


train_sizes, train_scores, test_scores = learning_curve(
    model_pipeline,
    X,
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


pickle.dump(model_pipeline, open("results/topic_classifier.pkl", "wb"))


if __name__ == "__main__":
    print("Training complete")