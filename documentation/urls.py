from django.urls import path

from . import views
from .views import markdown_uploader

urlpatterns = [
    path('', views.index, name='documentation'),
    path('api/uploader/', markdown_uploader, name='markdown_uploader_page'),
]
