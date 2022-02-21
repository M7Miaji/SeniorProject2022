# This for the RSS script that will be used to feed the news page
# This will also be used with conjunction to the the stock that the user has invested in

import feedparser
import webbrowser

# The feedparser that will get the application from the web 
feed = feedparser.parse("https://finance.yahoo.com/rss/")

# This is where every article will be read
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

# This will be used to make the access to the application much easier

def feedparser(id, title, summary, link):
    feed_entries = feed.entries 
    count = 0
    for entry in feed.entries:
        count = count + 1
        article_title =  entry.title
        article_link = entry.link 
        article_published_at = entry.published 
        article_published_at_parsed = entry.published_parsed 

        print(count, "- {}\n[{}]".format(article_title, article_link))
        print("Published at {}\n".format(article_published_at))

        return id, title, summary, link