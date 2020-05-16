from django.shortcuts import render, get_object_or_404

from project.models import Project
# Create your views here.
from topic_modelling.hdp_web import HDP
from topic_modelling.lda_web import LDA
from topic_modelling.lsa_web import LSA
from topic_modelling.nmf_web import NMF


def topic_algorithms(request, pk):
    content = {'pk': pk}
    return render(request, 'topic_modelling/index.html', content)


def apply_algorithm(request, pk, algorithm):
    project = get_object_or_404(Project, pk=pk)
    files = project.get_files()
    corpus = []
    for file in files:
        file = file.file
        file.open(mode='r')
        lines = file.read()
        file.close()
        corpus.append(lines)

    content = {}
    if algorithm.lower() == 'lda':
        content = LDA(corpus)
        content['algorithm'] = "LDA"
        content['project'] = project

    elif algorithm.lower() == 'lsa':
        content = LSA(corpus)
        content['algorithm'] = "LSA"
        content['project'] = project

    elif algorithm.lower() == 'hdp':
        content = HDP(corpus)
        content['algorithm'] = "HDP"
        content['project'] = project

    elif algorithm.lower() == 'nmf':
        content = NMF(corpus)
        content['algorithm'] = "NMF"
        content['project'] = project

    return render(request, 'topic_modelling/report.html', content)
