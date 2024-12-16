from django.urls import path
from . import views

urlpatterns = [
    path('', views.search_resumes, name='search_resumes'),

]