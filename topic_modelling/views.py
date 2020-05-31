import json

import plotly
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse

from project.models import Project
# Create your views here.
from topic_modelling.algorithms.hdp_web import HDP
from topic_modelling.algorithms.kmeans_web import w2v_kmeans, kmeans_optimum_value
from topic_modelling.algorithms.lda_web import LDA, lda_optimum_coherence
from topic_modelling.algorithms.lsa_web import LSA, lsa_optimum_coherence
from topic_modelling.algorithms.nmf_web import NMF, nmf_optimum_coherence
from topic_modelling.algorithms.topic_graph import tsne_graph_2d, tsne_graph_3d
from topic_modelling.models import Report


def topic_algorithms(request, pk):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project=project)

    content = {'project': project, 'reports': reports, 'title': f'Topic Modeling - {project.title}'}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Topic Modeling": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'topic_modelling/index.html', content)


def apply_topic_algorithm(request, pk, algorithm):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project_id=pk, algorithm=algorithm.lower())

    content = {'project': project, 'algorithm': algorithm, 'reports': reports,
               'title': f'{algorithm.upper()} - {project.title}'}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Topic Modeling": reverse('topic_algorithms', args=[pk]),
        algorithm.upper(): ""
        # algorithm.upper(): reverse('apply_topic_algorithm', args=[pk, algorithm])
    }

    content['breadcrumb'] = breadcrumb

    if request.method == 'POST':

        files = project.get_files()
        corpus = []
        for file in files:
            file = open(file.file.path, "r", encoding='utf8')
            lines = file.read()
            file.close()
            corpus.append(lines)

        if 'graph' in request.POST:
            start = int(request.POST['start'])
            end = int(request.POST['end'])
            step = int(request.POST['step'])

            if algorithm.lower() == 'lda':
                fig = lda_optimum_coherence(corpus, start, end, step)

            elif algorithm.lower() == 'lsa':
                fig = lsa_optimum_coherence(corpus, start, end, step)

            elif algorithm.lower() == 'nmf':
                fig = nmf_optimum_coherence(corpus, start, end, step)

            elif algorithm.lower() == 'w2v-kmeans':
                fig = kmeans_optimum_value(corpus, start, end, step)

            content["data"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render(request, 'topic_modelling/params.html', content)

        output = {}
        if 'n_topic' in request.POST:
            n_topic = int(request.POST['n_topic'])

        if algorithm.lower() == 'lda':
            output = LDA(corpus, n_topic)

        elif algorithm.lower() == 'lsa':
            output = LSA(corpus, n_topic)

        elif algorithm.lower() == 'hdp':
            output = HDP(corpus)

        elif algorithm.lower() == 'nmf':
            output = NMF(corpus, n_topic)

        elif algorithm.lower() == 'w2v-kmeans':
            output = w2v_kmeans(corpus, n_topic)

        content.update(output)
        content["files"] = [file.filename() for file in files]

        def my_converter(o):
            return o.__str__()

        report = Report()
        report.project = project
        report.algorithm = algorithm.lower()
        report.all_data = json.dumps(output, separators=(',', ':'), default=my_converter)
        report.topics = json.dumps(["Topic " + str(index + 1) for index in range(len(output['word_distributions']))])
        report.save()

        return redirect('view_report', project.id, algorithm, report.id)

    return render(request, 'topic_modelling/params.html', content)


def view_report(request, project_pk, algorithm, report_pk):
    project = get_object_or_404(Project, pk=project_pk)
    report = get_object_or_404(Report, pk=report_pk, algorithm=algorithm.lower())
    files = project.get_files()
    topics = report.get_topics()

    content = {
        'project': project,
        'algorithm': algorithm,
        'files': [file.filename() for file in files],
        'topics': topics,
        'report': report,
        'title': f'{algorithm.upper()} Report - {project.title}'
    }

    report_output = report.get_output()

    content.update(report_output)

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Topic Modeling": reverse('topic_algorithms', args=[project_pk]),
        algorithm.upper(): reverse('apply_topic_algorithm', args=[project_pk, algorithm]),
        f"Report (id:{report.id})": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'topic_modelling/report.html', content)


def get_graph(request, project_pk, algorithm, report_pk, graph_type):
    project = get_object_or_404(Project, pk=project_pk)
    report = get_object_or_404(Report, pk=report_pk, algorithm=algorithm.lower())
    files = project.get_files()
    topics = report.get_topics()
    report_output = report.get_output()

    response = ""
    if graph_type.lower() == 'graph_2d':
        response = tsne_graph_2d(report_output, topics, [file.filename() for file in files], algorithm)
    if graph_type.lower() == 'graph_3d':
        response = tsne_graph_3d(report_output, topics, [file.filename() for file in files], algorithm)

    return HttpResponse(response)


def set_report_topics(request, project_pk, algorithm, report_pk):
    if request.method == 'POST':
        report = get_object_or_404(Report, pk=report_pk, algorithm=algorithm.lower())
        topics = request.POST.getlist('topics[]')
        report.topics = json.dumps(topics)
        report.save()

    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))


def remove_report(request, project_pk, algorithm, report_pk):
    report = get_object_or_404(Report, pk=report_pk)
    report.delete()

    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
