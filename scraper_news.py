import time

import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd

def scrape_article(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("h1")
        if not title_tag:
            return None, None

        title = title_tag.get_text(strip=True)

        paragraphs = soup.find_all("p")

        clean_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True)

            # remove short lines
            if len(text) < 30:
                continue

            # remove BBC boilerplate
            if "BBC" in text and "listen" in text:
                continue

            clean_paragraphs.append(text)

        body = " ".join(clean_paragraphs)

        body = body.replace("\n", " ").strip()

        return title, body

    except Exception as e:
        return None, None

urls = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.bbci.co.uk/news/health/rss.xml",
    "https://feeds.bbci.co.uk/news/education/rss.xml",
    "https://feeds.bbci.co.uk/news/uk/rss.xml",
    "https://feeds.bbci.co.uk/news/us_and_canada/rss.xml",
    "https://feeds.bbci.co.uk/news/europe/rss.xml",
    "https://feeds.bbci.co.uk/news/asia/rss.xml",
    "https://feeds.bbci.co.uk/news/africa/rss.xml",
    "https://feeds.bbci.co.uk/news/middle_east/rss.xml"
]

feed_topics = {
    "https://feeds.bbci.co.uk/news/rss.xml": "general",
    "https://feeds.bbci.co.uk/news/world/rss.xml": "politics",
    "https://feeds.bbci.co.uk/news/business/rss.xml": "business",
    "https://feeds.bbci.co.uk/news/technology/rss.xml": "tech",
    "https://feeds.bbci.co.uk/news/politics/rss.xml": "politics",
    "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml": "entertainment",
    "https://feeds.bbci.co.uk/sport/rss.xml": "sport",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml": "tech",
    "https://feeds.bbci.co.uk/news/health/rss.xml": "tech",
    "https://feeds.bbci.co.uk/news/education/rss.xml": "politics",
    "https://feeds.bbci.co.uk/news/uk/rss.xml": "politics",
    "https://feeds.bbci.co.uk/news/us_and_canada/rss.xml": "politics",
    "https://feeds.bbci.co.uk/news/europe/rss.xml": "politics",
    "https://feeds.bbci.co.uk/news/asia/rss.xml": "politics",
    "https://feeds.bbci.co.uk/news/africa/rss.xml": "politics",
    "https://feeds.bbci.co.uk/news/middle_east/rss.xml": "politics"
}

article_id = 0
data = []

seen_urls = set()

for url in urls:
    feed = feedparser.parse(url)
    topic = feed_topics.get(url, "unknown")

    for entry in feed.entries:

        if entry.link in seen_urls:
            continue

        seen_urls.add(entry.link)
        
        title, body = scrape_article(entry.link)

        if title is None:
            continue

        data.append({
            "id": article_id,
            "url": entry.link,
            "date": entry.published,
            "headline": title,
            "body": body,
            "real_topic": topic
        })

        article_id += 1

        time.sleep(1)

        if len(data) >= 350:
            break
    
    if len(data) >= 350:
        break

df = pd.DataFrame(data)
df.to_csv("data/articles.csv", index=False)