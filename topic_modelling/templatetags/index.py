from django import template

register = template.Library()


@register.filter
def index(indexable, i):
    return indexable[int(i)]


@register.filter
def index_dict(indexable, i):
    return indexable[str(i)]
