# This for the RSS script that will be used to feed the news page
# This will also be used with conjunction to the the stock that the user has invested in

from multiprocessing.dummy import Array
import feedparser
import webbrowser

# The feedparser that will get the application from the web 
feed = feedparser.parse("https://www.cnbc.com/id/20409666/device/rss/rss.html?x=1")

# This is where every article will be read
def feedparser():
    feed_entries = feed.entries
    count = 0
    arr =output = [[0 for i in range(3)] for j in range(10)]

    for entry in feed.entries:
        count = count + 1
        article_title = entry.title
        article_link = entry.link
        article_published_at = entry.published 
        article_published_at_parsed = entry.published_parsed
        article_summary = entry.description
        arr[count-1][0]=article_title 
        arr[count-1][1]=article_published_at
        arr[count-1][2]=article_link
        print(article_summary)
        if count == 10:
            break
    data = {"title": arr}
    return data
    #print(count, "- {}\n[{}]".format(article_title, article_link))
    #print("Published at {}\n".format(article_published_at))

data = feedparser()
'''data1 = {"title": [["kyle", "Jan 1", "1999"],
                  ["kyle", "Jan 1", "1999"],
                   ["kyle", "Jan 1", "1999"]]}'''
#print(data)
# This will be used to make the access to the application much easier
"""
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
"""