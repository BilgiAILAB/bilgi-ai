import json

from django.db import models

# Create your models here.
from project.models import Project


class Report(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='topic_project')
    algorithm = models.CharField(max_length=100)
    all_data = models.TextField()
    topics = models.TextField()

    def get_output(self):
        return json.loads(self.all_data)

    def get_topics(self):
        return json.loads(self.topics)

    def coherence_value(self):
        try:
            return self.get_output()["coherence_value"]
        except KeyError:
            return None

    def silhouette_score(self):
        try:
            return self.get_output()["silhouette_score"]
        except KeyError:
            return None

    def topic_counts(self):
        return len(self.get_output()["word_distributions"])
