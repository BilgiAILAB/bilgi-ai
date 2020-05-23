from django.urls import path

from . import views

urlpatterns = [
    path('<int:pk>/topic-modelling', views.topic_algorithms, name='topic_algorithms'),
    path('<int:pk>/topic-modelling/<str:algorithm>', views.apply_topic_algorithm, name='apply_topic_algorithm'),
    path('<int:project_pk>/topic-modelling/<str:algorithm>/<int:report_pk>', views.view_report, name='view_report'),
    path('<int:project_pk>/topic-modelling/<str:algorithm>/<int:report_pk>', views.view_report, name='view_report'),
    path('<int:project_pk>/topic-modelling/<str:algorithm>/<int:report_pk>/remove', views.remove_report,
         name='remove_report'),
    path('<int:project_pk>/topic-modelling/<str:algorithm>/<int:report_pk>/topics', views.set_report_topics,
         name='set_topics'),
]
