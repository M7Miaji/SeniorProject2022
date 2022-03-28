# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.shortcuts import render
import feedparser
import webbrowser
from django.template.defaulttags import register

# The feedparser that will get the application from the web 
feed = feedparser.parse("https://www.cnbc.com/id/20409666/device/rss/rss.html?x=1")

@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        
        load_template = request.path.split('/')[-1]
        # Remove the if statment to get back at the orginal code
        if load_template == 'examples-project-detail.html':
            context = {}
            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            #context['segment'] = load_template
            context = feedparser()
            html_template = loader.get_template('home/' + load_template)
            return HttpResponse(html_template.render(context, request))
            
        elif load_template == 'tables-data.html':
            context = {
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            #context['segment'] = load_template
            html_template = loader.get_template('home/' + load_template)
            return HttpResponse(html_template.render(context, request))

        elif load_template == 'index3 copy.html':
            context = {
                "data": [22.5, 12, 18230.00, 33.1, 13, 12, 15, 29, 0.5, 28.2, 1230, 3, 1199, 199, 63, 352, 12, 0.8, 1],
            }
            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            #context['segment'] = load_template
            html_template = loader.get_template('home/' + load_template)
            return HttpResponse(html_template.render(context, request))
         
        else:
            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            context['segment'] = load_template

            html_template = loader.get_template('home/' + load_template)
            return HttpResponse(html_template.render(context, request))
    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))

def feedparser():
    
    feed_entries = feed.entries
    count = 0
    arr =output = [[0 for i in range(4)] for j in range(10)]

    for entry in feed.entries:
        count = count + 1
        article_title = entry.title
        article_link = entry.link
        article_published_at = entry.published 
        article_description = entry.description
        arr[count-1][0]=article_title 
        arr[count-1][1]=article_published_at
        arr[count-1][2]=article_link
        arr[count-1][3]=article_description
        if count == 10:
            break

    data = {"title": arr}
    return data

@register.filter
def get_value(dictionary, key):
    return dictionary.get(key)
