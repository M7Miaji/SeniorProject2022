# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.db import models

# Create your models here.

class Stocks(models.Model):
   Stock_id = models.IntegerField(max_length=500)
   stock_name = models.charField(max_length=100)
   Stock_price = models.DecimalField(max_length=1000)
    