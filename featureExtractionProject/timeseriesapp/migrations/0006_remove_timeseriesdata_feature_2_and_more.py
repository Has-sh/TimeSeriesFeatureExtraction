# Generated by Django 4.2.4 on 2023-08-15 18:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('timeseriesapp', '0005_rename_mean_timeseriesdata_feature_1_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='timeseriesdata',
            name='feature_2',
        ),
        migrations.AlterField(
            model_name='timeseriesdata',
            name='feature_1',
            field=models.JSONField(blank=True, null=True),
        ),
    ]