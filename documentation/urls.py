from django.urls import path

from .views import markdown_uploader

urlpatterns = [
    path('api/uploader/', markdown_uploader, name='markdown_uploader_page'),
]
