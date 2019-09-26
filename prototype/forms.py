from django import forms

class StepForm(forms.Form):
    step_count = forms.IntegerField( max_value= 2000, min_value= 5, label= 'enter step count:')