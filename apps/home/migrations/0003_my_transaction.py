# Generated by Django 3.2.11 on 2022-04-02 17:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_stock'),
    ]

    operations = [
        migrations.CreateModel(
            name='My_Transaction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mode', models.CharField(max_length=20)),
                ('company', models.CharField(max_length=50)),
                ('industry', models.CharField(max_length=50)),
                ('history', models.CharField(max_length=200)),
                ('profit_loss', models.CharField(max_length=15)),
            ],
        ),
    ]
