from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import StepForm
from .pylib.servers import messenger
from .pylib.servers.picam import new_receiver_thread
import os



def prototype(request):
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        tform = StepForm(request.POST)
        # check whether it's valid:
        if tform.is_valid():
            stepdata = tform.cleaned_data['step_count']
            print(stepdata)
            # messenger.proto_mess(stepdata)
            messenger.steps_mess(stepdata)
            # take_scan()
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


def take_scan():
    print('take scan')
    folder = '/home/samir/db3/prototype/static/scan_folder/scan_im_folder/'
    t = new_receiver_thread('1', folder=folder)
    print('scan receiver started')
    print('start take')
    # scan_mess()
    t.join()
    # folder = '/home/samir/db2/scan/static/scan_folder/scan_im_folder/'
    # # trim_files(folder)
    # scan_wrap(folder=folder)
    return 

# **********************************  NOt Used !!!! **********************************************



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

