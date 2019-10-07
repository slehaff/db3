from django.urls import path

from . import views

urlpatterns = [
    path('', views.prototype, name='prototype'),
#     path('testforms/', views.testforms, name='testforms'),
#    path('tests/', views.get_steps),
    
]
