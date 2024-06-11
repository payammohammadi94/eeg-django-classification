from django import forms

class ConfigForms(forms.Form):
    leadoff_mode = forms.BooleanField()
    sampling_rate = forms.IntegerField()
    
        