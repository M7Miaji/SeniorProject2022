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
import json
from django.template.defaulttags import register
from matplotlib.font_manager import json_dump
from .yahoo_api import get_quotes
from django.shortcuts import render
from .models import My_Transaction
from .models import Configuration
from .trade import main_lstm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from django.http import HttpResponse
from .macd import main_macd
from .bbands import main_bbands
from .sma import main_sma
from .ewma import main_ewma
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
            
        elif load_template == 'configuration.html':
            context = {}
            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            #context['segment'] = load_template
            html_template = loader.get_template('home/' + load_template)

            if request.method == 'POST':
                mode_ = request.POST['mode']
                industry_ = request.POST['industry']
                algorithm_ = request.POST['algorithm']
                risk_percentage_ = request.POST['risk_percentage']
                diversity_ = request.POST['diversity']
                max_buy_ = request.POST['max_buy']
                min_traded_ = request.POST['min_traded']
                max_traded_ = request.POST['max_traded']
                username_ = request.POST['username']
                username_ = request.user.username

                rows = Configuration.objects.filter(username=username_) 
                for r in rows: 
                    r.delete() 

                new_config = Configuration(mode=mode_, industry=industry_, algorithm=algorithm_, risk_percentage=risk_percentage_, diversity=diversity_, max_buy=max_buy_, min_traded=min_traded_, max_traded=max_traded_, username=username_)
                new_config.save()

                stock = ['AAPL']
                rows_alg = Configuration.objects.all()
                for i in range(len(rows_alg)):
                    if rows_alg[i].username == request.user.username:
                        industry_test = rows_alg[i].industry
                        algorithm_test = rows_alg[i].algorithm
                        risk_percentage_test = rows_alg[i].risk_percentage
                        diversity_test = rows_alg[i].diversity
                        max_buy_test = rows_alg[i].max_buy
                        min_traded_test = rows_alg[i].min_traded
                        max_traded_test = rows_alg[i].max_traded
                        
                        Buy, Sell, stock_info, signal = [0], [0], "", "Sell"
                        print(algorithm_test)
                        print("###################################################################################")
                        for i in stock:
                            try:
                                if algorithm_test == "SMA":
                                    Buy, Sell, stock_info, signal = main_sma(i)
                                elif algorithm_test == "EWMA":
                                    Buy, Sell, stock_info, signal = main_ewma(i)
                                elif algorithm_test == "BBANDS":
                                    Buy, Sell, stock_info, signal = main_bbands(i)
                                elif algorithm_test == "MACD":
                                    Buy, Sell, stock_info, signal = main_macd(i)
                                elif algorithm_test == "LSTM":
                                    array_per, array_org, accuracy, X_train, X_test, len_time, Next5Days, df= main_lstm(i)
                                
                                My_Transaction.objects.create(mode = algorithm_test, company = i, industry = 'TECH', history = signal, profit_loss = stock_info['Close'].iloc[-1], username = request.user.username)
                            except:
                                pass
                            
            return HttpResponse(html_template.render(context, request))

        elif load_template == 'tables-data.html': # My Transaction ADD POST----------------------------------------------------------
            context = {
            }
            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            #context['segment'] = load_template
            html_template = loader.get_template('home/' + load_template)

            all_data = My_Transaction.objects.all()
            usernames = request.user.username
            modes = []
            companys = []
            industries = []
            signals = []
            prices = []
            nums = []
            print(len(all_data))
            count = 1
            for i in range(len(all_data)):
                if all_data[i].username == usernames:
                    nums.append(count)
                    print("----",all_data[i].mode)
                    modes.append(all_data[i].mode)
                    companys.append(all_data[i].company)
                    industries.append(all_data[i].industry)
                    signals.append(all_data[i].history)
                    prices.append(all_data[i].profit_loss)
                    count = count + 1
            modes.reverse()
            companys.reverse()
            industries.reverse()
            signals.reverse()
            prices.reverse()
            mylist = zip(nums, modes, companys, signals, prices)
            print(modes)
            #print(modes)
            context = {
                "filename": "tables-data.html",
                "collapse": "",
                "mylist": mylist,
                "num": nums,
                "mode": modes,
                "company": companys,
                "industry": industries,
                "signal": signals,
                "price": prices,
            }

            return HttpResponse(html_template.render(context, request))
        
        elif load_template == 'charts-chartjs.html': # My Transaction ADD POST----------------------------------------------------------
    
            array_info = [0, 0, 0, 0]
            array = list(range(1, 201))
            array2 = list(range(1, 201))
            Next5Daysls = [0,0,0,0,0]

            if request.method == 'POST':
                search = request.POST['search']
                array = []
                array2 = []
                array_info = []
                Next5Daysls = []
                array_per, array_org, accuracy, X_train, X_test, len_time, Next5Days, df= main_lstm("AAPL")
                array_per.tolist()
                array_info = [X_train, X_test, len_time, accuracy]
                for i in range(5):
                    Next5Daysls.append(Next5Days[0][i])
                for i in range(200):
                    array.append(array_per[i]) 
                    array2.append(array_org[i])
            List = list(range(1, 201))
            for i in range(len(List)):
                List[i] = str(List[i])
            labels = List
            data = array
            data1 = array2
            context = {
                "filename": "charts-chartjs.html",
                "collapse": "",
                "labels": json.dumps(labels),
                "data1": json.dumps(data1),
                "data": json.dumps(data),
                "info": array_info,
                "next": Next5Daysls,
            }

            if load_template == 'admin':
                return HttpResponseRedirect(reverse('admin:index'))
            #context['segment'] = load_template
            html_template = loader.get_template('home/' + load_template)

            all_data = My_Transaction.objects.all()
            #for i in all_data:
            #    print(i.mode)
            return HttpResponse(html_template.render(context, request))

        elif load_template == 'index3 copy.html':
            search = 'AAPL'
            if request.method == 'POST':
                search = request.POST['search']
            data = get_quotes(search)
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

