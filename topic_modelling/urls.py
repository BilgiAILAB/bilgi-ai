from django.urls import path

from . import views

urlpatterns = [
    path('<int:pk>', views.topic_algorithms, name='topic_algorithms'),
    path('<int:pk>/lda', views.apply_lda, name='apply_lda'),
]