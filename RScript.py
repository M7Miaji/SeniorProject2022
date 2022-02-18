import feedparser
import webbrowser

feed = feedparser.parse("https://finance.yahoo.com/rss/")

feed_entries = feed.entries
count = 0
for entry in feed.entries:
    count = count + 1
    article_title = entry.title
    article_link = entry.link
    article_published_at = entry.published 
    article_published_at_parsed = entry.published_parsed

    print(count, "- {}\n[{}]".format(article_title, article_link))
    print("Published at {}\n".format(article_published_at))