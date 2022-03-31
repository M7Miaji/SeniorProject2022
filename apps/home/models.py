
from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=40)
    email = models.EmailField(max_length=100)
    pwd = models.CharField(max_length=40)
    favstocks= models.CharField(max_length=15000)
 


class Stock(models.Model):
   StockName=models.CharField(max_length=100)
   stockprice=models.DecimalField()

    

