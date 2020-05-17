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
    content = {'project': project, 'algorithm': algorithm}

    if request.method == 'POST':

        n_topic = int(request.POST['n_topic'])

        files = project.get_files()
        corpus = []
        for file in files:
            file = open(file.file.path, "r", encoding='utf8')
            lines = file.read()
            file.close()
            corpus.append(lines)

        output = {}
        if algorithm.lower() == 'lda':
            output = LDA(corpus, n_topic)

        elif algorithm.lower() == 'lsa':
            output = LSA(corpus, n_topic)

        elif algorithm.lower() == 'hdp':
            output = HDP(corpus, n_topic)

        elif algorithm.lower() == 'nmf':
            output = NMF(corpus, n_topic)

        content.update(output)
        content["files"] = [file.filename() for file in files]

        return render(request, 'topic_modelling/report.html', content)

    return render(request, 'topic_modelling/params.html', content)
