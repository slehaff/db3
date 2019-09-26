from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import StepForm


def prototype(request):
    context = {"prototype_page": "active"}
    return render(request, 'prototype.html', context)


def testforms(request):
    context = {"testform_page": "active"}
    return render(request, 'testforms.html', context)


def gettest(request):
    print('got it!!!')


def get_steps(request):
    print('getting steps!!')
    if request.method == 'POST':
        form = StepForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('/thanks/')
    else:
        form = StepForm()
    
    return render(request, 'testform.html', {'form': form})