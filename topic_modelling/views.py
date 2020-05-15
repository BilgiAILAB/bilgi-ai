from django.shortcuts import render, get_object_or_404

# Create your views here.
from lda_web import LDA
from project.models import Project


def topic_algorithms(request, pk):
    content = {'pk': pk}
    return render(request, 'topic_modelling/index.html', content)


def apply_lda(request, pk):
    project = get_object_or_404(Project, pk=pk)
    files = project.get_files()
    corpus = []
    for file in files:
        file = file.file
        file.open(mode='r')
        lines = file.read()
        file.close()
        corpus.append(lines)
    content = LDA(corpus)
    content['project'] = project
    return render(request, 'topic_modelling/lda.html', content)
