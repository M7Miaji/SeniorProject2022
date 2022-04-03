
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
    mode=models.CharField(max_length=20)
    company=models.CharField(max_length=50)
    industry=models.CharField(max_length=50)
    history=models.CharField(max_length=200)
    profit_loss=models.IntegerField()

