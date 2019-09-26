from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import StepForm


def prototype(request):
    context = {"prototype_page": "active"}
    return render(request, 'prototype.html', context)


def testforms(request):
    context = {"testform_page": "active"}
    return render(request, 'testforms.html', context)


def get_steps(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = StepForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = StepForm()

    return render(request, 'testforms.html', {'form': form})