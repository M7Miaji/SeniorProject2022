# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib import admin

from .models import My_Transaction, User
from .models import Stock
# Register your models here.
admin.site.register(User)
admin.site.register(Stock)
admin.site.register(My_Transaction)