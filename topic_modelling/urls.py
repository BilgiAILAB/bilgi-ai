from django.urls import path

from . import views

urlpatterns = [
    path('<int:pk>', views.topic_algorithms, name='topic_algorithms'),
    path('<int:pk>/<str:algorithm>', views.apply_algorithm, name='apply_algorithm'),
    path('<int:project_pk>/<str:algorithm>/<int:report_pk>', views.view_report, name='view_report'),
    path('<int:project_pk>/<str:algorithm>/<int:report_pk>', views.view_report, name='view_report'),
    path('<int:project_pk>/<str:algorithm>/<int:report_pk>/remove', views.remove_report, name='remove_report'),
]
