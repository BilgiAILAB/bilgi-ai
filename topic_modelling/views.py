import json

from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse

from project.models import Project
# Create your views here.
from topic_modelling.hdp_web import HDP
from topic_modelling.lda_web import LDA
from topic_modelling.lsa_web import LSA
from topic_modelling.models import Report
from topic_modelling.nmf_web import NMF


def topic_algorithms(request, pk):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project=project)

    content = {'project': project, 'reports': reports}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Topic Modelling": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'topic_modelling/index.html', content)


def apply_algorithm(request, pk, algorithm):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project_id=pk, algorithm=algorithm.lower())

    content = {'project': project, 'algorithm': algorithm, 'reports': reports}

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

        def my_converter(o):
            return o.__str__()

        report = Report()
        report.project = project
        report.algorithm = algorithm.lower()
        report.all_data = json.dumps(output, separators=(',', ':'), default=my_converter)
        report.save()

        return redirect('view_report', project.id, algorithm, report.id)

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Topic Modelling": reverse('topic_algorithms', args=[pk]),
        algorithm.upper(): ""
        # algorithm.upper(): reverse('apply_algorithm', args=[pk, algorithm])
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'topic_modelling/params.html', content)


def view_report(request, project_pk, algorithm, report_pk):
    project = get_object_or_404(Project, pk=project_pk)
    report = get_object_or_404(Report, pk=report_pk, algorithm=algorithm.lower())
    files = project.get_files()

    content = {'project': project,
               'algorithm': algorithm,
               "files": [file.filename() for file in files]}

    content.update(report.get_output())

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Topic Modelling": reverse('topic_algorithms', args=[project_pk]),
        algorithm.upper(): reverse('apply_algorithm', args=[project_pk, algorithm]),
        f"Report (id:{report.id})": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'topic_modelling/report.html', content)


def remove_report(request, project_pk, algorithm, report_pk):
    report = get_object_or_404(Report, pk=report_pk)
    report.delete()

    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
