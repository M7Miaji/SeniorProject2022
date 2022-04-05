# Generated by Django 3.2.11 on 2022-04-05 12:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0003_my_transaction'),
    ]

    operations = [
        migrations.CreateModel(
            name='Configuration',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mode', models.CharField(max_length=20)),
                ('industry', models.CharField(max_length=50)),
                ('algorithm', models.CharField(max_length=50)),
                ('risk_percentage', models.IntegerField()),
                ('diversity', models.IntegerField()),
                ('max_buy', models.IntegerField()),
                ('min_traded', models.IntegerField()),
                ('max_traded', models.IntegerField()),
                ('username', models.CharField(max_length=50)),
            ],
        ),
        migrations.AlterField(
            model_name='my_transaction',
            name='profit_loss',
            field=models.IntegerField(),
        ),
    ]
