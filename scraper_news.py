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

        title = title_tag.text.strip()
        paragraphs = soup.find_all("p")
        body = " ".join([p.text for p in paragraphs])

        return title, body[:500]

    except:
        return None, None
    

urls = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",

    "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
    "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
    "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",

    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.bbci.co.uk/news/health/rss.xml",
    "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"
]

article_id = 0
data = []

seen_urls = set()

for url in urls:
    feed = feedparser.parse(url)

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
            "body": body
        })

        article_id += 1

        time.sleep(1)

df = pd.DataFrame(data)
df.to_csv("data/articles.csv", index=False)