from django import forms

class CalibrateForm(forms.Form):
    calibration_type = forms.IntegerField( max_value= 100, min_value= 1, label= 'Calibration Folder Number:')

    def clean_calibration_type(self):
        data = self.cleaned_data['calibration_type']
        
        # Check if a date is not in the past. 
        print("Calibrate!!!")

        # Remember to always return the cleaned data.
        return data