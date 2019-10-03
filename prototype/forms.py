from django import forms

class StepForm(forms.Form):
    step_count = forms.IntegerField( max_value= 2000, min_value= 5, label= 'enter step count:')
    voltage = forms.IntegerField(max_value= 3, min_value= .2, label= 'enter voltage:')
    def clean_stepcount(self):
        data = self.cleaned_data['step_count']
        
        # Check if a date is not in the past. 
        print("It's clean!!!")

        # Remember to always return the cleaned data.
        return data


    def clean_voltage(self):
        volts = self.cleaned_data['voltage']
        
        # Check if a date is not in the past. 
        print("It's clean!!!")

        # Remember to always return the cleaned data.
        return volts