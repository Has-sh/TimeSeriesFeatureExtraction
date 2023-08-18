from django import forms
from .models import TimeSeriesData

class TimeSeriesDataForm(forms.ModelForm):
    csv_file = forms.FileField()

    class Meta:
        model = TimeSeriesData
        fields = ['csv_file']
