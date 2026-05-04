# 📰 NLP News Scraper & Analysis Pipeline

## 📌 Project Overview

This project scrapes news articles from BBC RSS feeds and builds a full NLP pipeline to analyze them. It performs topic classification, sentiment analysis, organization extraction, and environmental scandal detection using machine learning and NLP techniques.

---

## ⚙️ Features

* 📥 News scraping from BBC RSS feeds
* 🧹 Text cleaning and preprocessing
* 🏷️ Topic classification (Tech, Sport, Business, Politics, Entertainment)
* 😊 Sentiment analysis using VADER
* 🏢 Named Entity Recognition (Organizations extraction with spaCy)
* 🌍 Environmental scandal detection using sentence embeddings
* 📊 Top-10 scandal ranking system

---

## 🧠 Machine Learning Model

* Model: Linear Support Vector Classifier (LinearSVC)
* Text Vectorization: TF-IDF (1–2 grams)
* Accuracy:

  * BBC dataset: ~98%
  * Real scraped data: ~75%

Saved model:

```
results/topic_classifier.pkl
```

---

## 📂 Project Structure

```
data/
  bbc_news_train.csv
  bbc_news_tests.csv
  articles.csv

results/
  topic_classifier.pkl
  training_model.py
  enhanced_news.csv
  learning_curves.png

scraper_news.py
nlp_enriched_news.py
README.md
requirements.txt
setup_nltk.py
helper.sh
```

---

## 🚀 How to Run

### 1. Scrape articles

```bash
python3 scraper_news.py
```

### 2. Train topic classifier

```bash
python3 results/training_model.py
```

### 3. Run NLP pipeline

```bash
python3 nlp_enriched_news.py
```

---

## 📊 Output Columns

* id
* url
* date
* headline
* body
* orgs
* topics
* sentiment
* scandal_score
* top_10
* is_scandal

---

## 🌍 Scandal Detection

This part of the project detects environmental scandals in news articles using **sentence embeddings** and **cosine similarity**.

We use a pre-trained transformer model (**all-MiniLM-L6-v2**) to convert both text and keywords into numerical vectors (embeddings). This allows the model to understand the meaning of words, not just exact matches.

Cosine similarity is used to measure how close each sentence is to environmental concepts. A higher score means the article is more related to an environmental scandal.

Only the highest similarity score per article is kept as the final scandal score.

We use cosine similarity, so we name it scandal_score (semantic similarity score).

---

### 🔑 Keywords used

* pollution
* oil spill
* toxic waste
* deforestation
* emissions
* chemical spill
* contamination
* hazardous waste
* sewage
* environmental damage

---

### 📊 Output

* `scandal_score`: similarity score between article and keywords
* `top_10`: True if the article is in the top 10 highest scandal scores

---

### 🚨 Top 10 selection

```python
df["top_10"] = df["scandal_score"] >= df["scandal_score"].nlargest(10).min()
```

---

### 🧠 Why embeddings and cosine similarity?

We use embeddings because they capture the meaning of text instead of exact words. This helps detect environmental scandals even when different vocabulary is used.

Cosine similarity is used because it gives a normalized score (0 to 1) that represents how semantically close two texts are.

---

## 📈 Learning Curve

A learning curve is generated to evaluate overfitting and model performance:

```
results/learning_curves.png
```