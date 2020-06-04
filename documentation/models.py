from django.db import models
from martor.models import MartorField


class MainTopic(models.Model):
    name = models.CharField(max_length=256)

    def documentations(self):
        return DocumentationContent.objects.filter(main=self)

    def __str__(self):
        return self.name


class DocumentationContent(models.Model):
    name = models.CharField(max_length=256)
    main = models.ForeignKey(MainTopic, on_delete=models.CASCADE)
    content = MartorField()

    def __str__(self):
        return self.main.name + " " + self.name
