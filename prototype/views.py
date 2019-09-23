from django.shortcuts import render
from django.http import HttpResponse


def prototype(request):
    return render(request, 'prototype.html', {})