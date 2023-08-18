from django.contrib import admin
from .models import TimeSeriesData

class TimeSeriesDataAdmin(admin.ModelAdmin):
    list_display = ('id','timestamp', 'features')  # Display these fields in the list view
    
# Register your model with the admin site using the custom admin class
admin.site.register(TimeSeriesData, TimeSeriesDataAdmin)
