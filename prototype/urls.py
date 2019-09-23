from django.urls import path

from . import views

urlpatterns = [
    path('', views.prototype, name='prototype'),
]
