# Generated by Django 3.2.11 on 2022-04-08 14:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0005_auto_20220406_1529'),
    ]

    operations = [
        migrations.AlterField(
            model_name='my_transaction',
            name='username',
            field=models.CharField(max_length=50),
        ),
    ]
