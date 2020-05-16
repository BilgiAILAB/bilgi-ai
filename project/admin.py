from django.contrib import admin

# Register your models here.
from project.models import ProjectFile, Project

admin.site.register(ProjectFile)
admin.site.register(Project)
