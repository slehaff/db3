from django.urls import path

from . import views

urlpatterns = [
    path('', views.calibrate, name='calibrate'),
#     path('testforms/', views.testforms, name='testforms'),
#    path('tests/', views.get_steps),
    
]