from django.shortcuts import render, get_object_or_404, redirect

from project.models import Project
# Create your views here.
from topic_modelling.hdp_web import HDP
from topic_modelling.lda_web import LDA
from topic_modelling.lsa_web import LSA
from topic_modelling.nmf_web import NMF


def topic_algorithms(request, pk):
    content = {'pk': pk}
    return render(request, 'topic_modelling/index.html', content)


def get_params_before_apply_algorithm(request, pk, algorithm):
    if request.method == 'POST':
        redirect('apply_algorithm', pk, algorithm, )


def apply_algorithm(request, pk, algorithm):
    project = get_object_or_404(Project, pk=pk)
    files = project.get_files()
    corpus = []
    for file in files:
        print(file.file.name)
        file = open(file.file.path, "r", encoding='utf8')
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

    content["files"] = files

    return render(request, 'topic_modelling/report.html', content)
