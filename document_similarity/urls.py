from django.urls import path

from . import views

urlpatterns = [
    path('<int:pk>/document-similarity', views.similarity_algorithms, name='similarity_algorithms'),
    path('<int:pk>/document-similarity/<str:algorithm>', views.apply_similarity_algorithm,
         name='apply_similarity_algorithm'),
    path('<int:project_pk>/document-similarity/<str:algorithm>/<int:report_pk>', views.view_similarity_report,
         name='view_similarity_report'),
    path('<int:project_pk>/document-similarity/<str:algorithm>/<int:report_pk>/remove', views.remove_similarity_report,
         name='remove_similarity_report'),

]
