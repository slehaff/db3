from django.urls import path

from . import views

urlpatterns = [
    path('', views.scan, name='scan'),
#     path('testforms/', views.testforms, name='testforms'),
   path('testhttp/', views.get_http),   
]
