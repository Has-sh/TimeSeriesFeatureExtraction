# Generated by Django 4.2.4 on 2023-08-15 17:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('timeseriesapp', '0003_timeseriesdata_timestamp_alter_timeseriesdata_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='timeseriesdata',
            name='value',
            field=models.FloatField(default=0),
        ),
    ]