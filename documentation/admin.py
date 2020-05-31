from django.contrib import admin
from django.db import models
# Register your models here.
from martor.widgets import AdminMartorWidget

from documentation.models import DocumentationContent, MainTopic


class DocumentationContentModelAdmin(admin.ModelAdmin):
    formfield_overrides = {
        models.TextField: {'widget': AdminMartorWidget}
    }


admin.site.register(MainTopic)
admin.site.register(DocumentationContent, DocumentationContentModelAdmin)
