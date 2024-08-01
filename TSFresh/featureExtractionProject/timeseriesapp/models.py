from django.db import models

class TimeSeriesData(models.Model):
    timestamp = models.CharField(max_length=255,default=0)
    features = models.JSONField(blank=True, null=True)

    def __str__(self):
        return str(self.timestamp)