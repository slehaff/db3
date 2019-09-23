from django.shortcuts import render
from django.http import HttpResponse


def prototype(request):
    context = {"prototype_page": "active"}
    return render(request, 'prototype.html', context)