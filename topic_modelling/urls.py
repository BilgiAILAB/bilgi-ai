from django.urls import path

from . import views

urlpatterns = [
    path('<int:pk>/topic-modeling', views.topic_algorithms, name='topic_algorithms'),
    path('<int:pk>/topic-modeling/<str:algorithm>', views.apply_topic_algorithm, name='apply_topic_algorithm'),
    path('<int:project_pk>/topic-modeling/<str:algorithm>/<int:report_pk>', views.view_report, name='view_report'),
    path('<int:project_pk>/topic-modeling/<str:algorithm>/<int:report_pk>/graph/<str:graph_type>', views.get_graph,
         name='get_graph'),
    path('<int:project_pk>/topic-modeling/<str:algorithm>/<int:report_pk>/remove', views.remove_report,
         name='remove_report'),
    path('<int:project_pk>/topic-modeling/<str:algorithm>/<int:report_pk>/topics', views.set_report_topics,
         name='set_topics'),
]
