from django import forms

class ScanForm(forms.Form):
    scan_type = forms.IntegerField( max_value= 5, min_value= 1, label= 'select scan type:', required=False)

    def clean_scan_type(self):
        data = self.cleaned_data['scan_type']
        
        # Check if a date is not in the past. 
        print("Go Scan!!!")

        # Remember to always return the cleaned data.
        return data