import pandas as pd
import spacy
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/articles.csv")

# ORG EXTRACTION
nlp = spacy.load("en_core_web_sm")

docs = nlp.pipe(df["body"].fillna(""), batch_size=20)

orgs_list = []
for doc in docs:
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    orgs_list.append(list(set(orgs)))

df["orgs"] = orgs_list


# SENTIMENT
sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return sia.polarity_scores(str(text))["compound"]

df["sentiment"] = df["body"].apply(get_sentiment_score)


# TOPIC CLASSIFICATION
model_clf = pickle.load(open("results/topic_classifier.pkl", "rb"))
vectorizer = pickle.load(open("results/vectorizer.pkl", "rb"))

def clean_text(text):
    return str(text).lower().strip()

df["clean_body"] = df["body"].apply(clean_text)

X_vec = vectorizer.transform(df["clean_body"])
df["topics"] = model_clf.predict(X_vec)


# SCANDAL DETECTION
model = SentenceTransformer("all-MiniLM-L6-v2")

keywords = [
    "pollution",
    "oil spill",
    "deforestation",
    "environmental disaster",
    "toxic waste",
    "climate damage"
]

keyword_embeddings = model.encode(keywords)

def scandal_score(text):
    sentences = str(text).split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    sentence_embeddings = model.encode(sentences)
    similarities = cosine_similarity(sentence_embeddings, keyword_embeddings)

    return similarities.max()

df["scandal_score"] = df["body"].apply(scandal_score)


# TOP 10 FLAG
threshold = df["scandal_score"].nlargest(10).min()
df["top_10"] = df["scandal_score"] >= threshold


# SAVE
df.to_csv("results/enhanced_news.csv", index=False)