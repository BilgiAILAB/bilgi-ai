from django.urls import path

from . import views
from .views import markdown_uploader

urlpatterns = [
    path('index', views.index, name='documentation'),
    path('show/<int:pk>', views.show_documentation, name='show_documentation'),
    path('api/uploader/', markdown_uploader, name='markdown_uploader_page'),
]
