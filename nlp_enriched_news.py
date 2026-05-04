import pandas as pd
import spacy
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/articles.csv")

# CLEAN DATA AND REMOVE WEAK Artivles
df = df.dropna(subset=["body"])
df = df[df["body"].str.len() > 200]
df["body"] = df["body"].fillna("")

# ORG EXTRACTION
nlp = spacy.load("en_core_web_sm")

docs = nlp.pipe(df["body"], batch_size=20)

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
model = pickle.load(open("results/topic_classifier.pkl", "rb"))
df["topics"] = model.predict(df["body"])


# SCANDAL DETECTION
model = SentenceTransformer("all-MiniLM-L6-v2")

keywords = [
    "pollution", "water pollution", "air pollution",
    "oil spill", "chemical spill",
    "toxic waste", "hazardous waste",
    "deforestation", "illegal logging",
    "carbon emissions", "climate damage",
    "industrial contamination", "environmental disaster"
]

keyword_embeddings = model.encode(keywords)

def scandal_score(text):
    sentences = str(text).split(".")
    sentences = [s.strip() for s in sentences]

    sentence_embeddings = model.encode(sentences)
    similarities = cosine_similarity(sentence_embeddings, keyword_embeddings)

    # return similarities.max()
    top_scores = similarities.max(axis=1)
    top_scores = sorted(top_scores, reverse=True)[:3]

    return sum(top_scores) / len(top_scores)

df["scandal_score"] = df["body"].apply(scandal_score)
df["is_scandal"] = df["scandal_score"] > 0.45
df["scandal_distance"] = 1 - df["scandal_score"]

# TOP 10 FLAG
threshold = df["scandal_score"].nlargest(10).min()
df["top_10"] = df["scandal_score"] >= threshold

# LOGS
for i, row in df.iterrows():

    url = row.get("url", "")
    body = row["body"]
    orgs = row["orgs"]
    topic = row["topics"]
    sentiment = row["sentiment"]
    scandal_score = row["scandal_score"]

    print(f"\nEnriching {url}")

    print("---------- Detect entities ----------")
    print(f"Detected {len(orgs)} companies: {orgs}")

    print("---------- Topic detection ----------")
    print(f"The topic of the article is: {topic}")

    print("---------- Sentiment analysis ----------")
    print(f"The article has a sentiment of {sentiment}")

    print("---------- Scandal detection ----------")
    print(f"Environmental scandal score: {scandal_score}")

# SAVE
df.to_csv("results/enhanced_news.csv", index=False)