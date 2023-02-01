from django import forms
from .models import person, login_data

class TestForm(forms.Form):
    id = forms.CharField(label = "user_id")

class LoginForm(forms.Form):
    class Meta:
        pass

class CreateUserForm(forms.ModelForm):
    class Meta:
        pass
        model=person
        fields = ("user_name", "user_id")
