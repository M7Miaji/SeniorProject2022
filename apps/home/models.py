# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import email
from pyexpat import model
from django.db import models
from django.contrib.auth.models import User

# Create your models here.

Class Members(models.Model):
    fname = models.CharField(max_length=40)
    email = models.EmailField(max_length=100)
    pwd = models.charField(max_length=40)
    favstocks=models.TextField(max_length=15000)

