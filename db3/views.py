from django.http import HttpResponse

def index(request):
    return HttpResponse('index view from main!')