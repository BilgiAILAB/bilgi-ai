import os

from django.db import models

# Create your models here.
from django.utils.text import slugify


class Project(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField(max_length=1024)
    project_folder = models.CharField(max_length=32)
    date = models.DateTimeField(auto_now_add=True)

    def get_files(self):
        return ProjectFile.objects.filter(project=self)

    def save(self, *args, **kwargs):
        self.project_folder = slugify(self.title)
        super(Project, self).save(*args, **kwargs)


class ProjectFile(models.Model):
    def get_project_folder(self, filename):
        return f'data/projects/{self.project.project_folder}/{filename}'

    def filename(self):
        return os.path.basename(self.file.name)

    def __str__(self):
        return f'{self.id} {self.file.name} {self.project.title}'

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    file = models.FileField(upload_to=get_project_folder)
