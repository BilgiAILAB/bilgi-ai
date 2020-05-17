import json

from django.db import models

# Create your models here.
from project.models import Project


class Report(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    algorithm = models.CharField(max_length=100)
    all_data = models.TextField()

    def get_output(self):
        return json.loads(self.all_data)

    def coherence(self):
        return self.get_output()["coherence_value"]

    def topic_counts(self):
        return len(self.get_output()["word_distributions"])
