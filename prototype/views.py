from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import StepForm
from .pylib import messenger


def prototype(request):
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        tform = StepForm(request.POST)
        # check whether it's valid:
        if tform.is_valid():
            stepdata = tform.cleaned_data['step_count']
            messenger.proto_mess(stepdata)
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            tform = StepForm()
            # return HttpResponse('/thanks/')
    else:
        'empty form'
        tform = StepForm()
    context = {"prototype_page": "active", 'tform': tform}
    return render(request, 'prototype.html', context)


def testforms(request):
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        tform = StepForm(request.POST)
        # check whether it's valid:
        if tform.is_valid():
            print(tform.cleaned_data)
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            tform = StepForm()
            # return HttpResponse('/thanks/')
    else:
        tform = StepForm()
    return render(request, 'testforms.html', {'tform': tform})


def get_steps(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = StepForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            print(form.cleaned_data)
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = StepForm()

    return render(request, 'testforms.html', {'form': form})