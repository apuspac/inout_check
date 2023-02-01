from django.contrib import admin
from .models import login_data, person

# Register your models here.
admin.site.register(login_data)
admin.site.register(person)