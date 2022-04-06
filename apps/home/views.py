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
from .yahoo_api import get_quotes
from django.shortcuts import render
from .models import My_Transaction
from .models import Configuration

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
            
        elif load_template == 'configuration.html': # My Transaction ADD POST----------------------------------------------------------
            context = {}
            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            #context['segment'] = load_template
            html_template = loader.get_template('home/' + load_template)

            if request.method == 'POST':
                '''mode_ = request.method['mode']
                industry_ = request.method['industry']
                algorithm_ = request.method['algorithm']
                risk_percentage_ = request.method['risk_percentage']
                diversity_ = request.method['diversity']
                max_buy_ = request.method['max_buy']
                min_traded_ = request.method['min_traded']
                max_traded_ = request.method['max_traded']
                username_ = request.method['username']'''

                mode_ = 'Automatic'
                industry_ = 'Cars'
                algorithm_ = 'Mean'
                risk_percentage_ = '12%' 
                diversity_ = '1-2'
                max_buy_ = 1200
                min_traded_ = 10
                max_traded_ = 23
                username_ = 'admin'

                new_config = Configuration(mode=mode_, industry=industry_, algorithm=algorithm_, risk_percentage=risk_percentage_, diversity=diversity_, max_buy=max_buy_, min_traded=min_traded_, max_traded=max_traded_, username=username_)
                new_config.save()

            return HttpResponse(html_template.render(context, request))

        elif load_template == 'tables-data.html': # My Transaction ADD POST----------------------------------------------------------
            context = {
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            #context['segment'] = load_template
            html_template = loader.get_template('home/' + load_template)
            return HttpResponse(html_template.render(context, request))

        elif load_template == 'index3 copy.html':
            data = get_quotes("AAPL")
            context = {
                "data": [data['quoteResponse']['result'][0]['regularMarketPrice'], data['quoteResponse']['result'][0]['regularMarketChangePercent'], data['quoteResponse']['result'][0]['marketCap'], 33.1, data['quoteResponse']['result'][0]['trailingPE'], 12, 15, data['quoteResponse']['result'][0]['priceToSales'], 0.5, 28.2, data['quoteResponse']['result'][0]['revenue'], 3, 1199, data['quoteResponse']['result'][0]['pegRatio'], 63, 352, 12, 0.8, 1],
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


# Assuming the data to be entered is presnet in these lists
tran_mode = ['Automatic', 'Manual']
tran_company = ['Apple', 'Amazon']
tran_industry = ['Tech', 'Books']
tran_history = ['Trending Algorithm', 'Mean Algorithm']
tran_profit_loss = [20, 15]

def my_view(request, *args, **kwargs):
    
    # Iterate through all the data items
    for i in range(len(tran_mode)):

        # Insert in the database
        My_Transaction.objects.create(Mode = tran_mode[i], Company = tran_company[i], Industry = tran_industry[i], History = tran_history[i], Profit_Loss = tran_profit_loss[i])


    # Getting all the stuff from database
    query_results = My_Transaction.objects.all()

    # Creating a dictionary to pass as an argument
    context = { 'query_results' : query_results }

    # Returning the rendered html
    return render(request, "home.html", context)