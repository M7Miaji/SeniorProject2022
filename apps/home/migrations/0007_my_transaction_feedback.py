# Generated by Django 3.2.11 on 2022-04-09 20:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0006_alter_my_transaction_username'),
    ]

    operations = [
        migrations.AddField(
            model_name='my_transaction',
            name='feedback',
            field=models.CharField(default=0, max_length=300),
            preserve_default=False,
        ),
    ]