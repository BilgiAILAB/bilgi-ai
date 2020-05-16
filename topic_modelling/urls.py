from django.urls import path

from . import views

urlpatterns = [
    path('<int:pk>', views.topic_algorithms, name='topic_algorithms'),
    path('<int:pk>/<str:algorithm>', views.apply_algorithm, name='apply_algorithm'),
]
