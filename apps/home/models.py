
from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=40)
    email = models.EmailField(max_length=100)
    pwd = models.CharField(max_length=40)
    favstocks= models.CharField(max_length=15000)
 


class Stock(models.Model):
   StockName=models.CharField(max_length=100)
   

class My_Transaction(models.Model):
    hash=models.BigIntegerField(max_length=100000)
    mode=models.CharField(max_length=20)
    company=models.CharField(max_length=50)
    industry=models.CharField(max_length=50)
    history=models.CharField(max_length=200)
    profit_loss=models.IntegerField()
    username=models.CharField(max_length=50)
    


class Configuration(models.Model):
    mode=models.CharField(max_length=20)
    industry=models.CharField(max_length=50)
    algorithm=models.CharField(max_length=50)
    risk_percentage=models.CharField(max_length=10)
    diversity=models.CharField(max_length=10)
    max_buy=models.IntegerField()
    min_traded=models.IntegerField()
    max_traded=models.IntegerField()
    username=models.CharField(max_length=50)