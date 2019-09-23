from django.http import HttpResponse
from django.shortcuts import render

context = {"home_page": "active"}
def index(request):
    return render(request, 'main.html', context)