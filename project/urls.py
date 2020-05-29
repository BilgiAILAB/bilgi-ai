"""thesis_django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from . import views

urlpatterns = [
    path('', views.all_projects, name='all_projects'),
    path('new', views.create_project, name='create_project'),
    path('<int:pk>', views.show_project, name='show_project'),
    path('<int:pk>/remove', views.delete_project, name='delete_project'),
    path('<int:pk>/upload', views.add_files, name='add_files'),
    path('<int:pk>/download', views.download, name='download_files'),
    path('<int:file_pk>/full-view-to-file', views.full_view_to_file, name='full_view_to_file'),

]
