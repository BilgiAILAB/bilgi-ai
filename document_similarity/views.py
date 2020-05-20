import json

from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse

from document_similarity.models import Report
from document_similarity.similarity_algorithms import documentsCosineSimilarity, documentsEuclideanDistance, \
    documentsJaccardSimilarity, documentsManhattanDistance
from project.models import Project


def similarity_algorithms(request, pk):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project_id=pk)

    content = {'project': project, 'reports': reports}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Document Similarity": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'document_similarity/index.html', content)


def apply_similarity_algorithm(request, pk, algorithm):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project_id=pk, algorithm=algorithm.lower())

    content = {'project': project, 'algorithm': algorithm, 'reports': reports, 'files': project.get_files()}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Document Similarity": reverse('similarity_algorithms', args=[pk]),
        algorithm.upper(): ""
    }

    content['breadcrumb'] = breadcrumb

    if request.method == 'POST':

        selected_file_id = int(request.POST['file'])
        files = project.get_files()
        corpus = []
        index = 0
        selected_document_index = 0
        selected_document_name = 0
        for file in files:
            if file.id == selected_file_id:
                selected_document_index = index
                selected_document_name = file.filename()
            index += 1

            file_read = open(file.file.path, "r", encoding='utf8')
            lines = file_read.read()
            file_read.close()
            corpus.append(lines)

        if algorithm.lower() == 'tfidf-cos':
            outputs = documentsCosineSimilarity(selected_document_index, corpus)
        elif algorithm.lower() == 'tfidf-euc':
            outputs = documentsEuclideanDistance(selected_document_index, corpus)
        elif algorithm.lower() == 'tfidf-jac':
            outputs = documentsJaccardSimilarity(selected_document_index, corpus)
        elif algorithm.lower() == 'tfidf-man':
            outputs = documentsManhattanDistance(selected_document_index, corpus)

        content['outputs'] = outputs
        content['selected_document_index'] = selected_document_index

        report = Report()
        report.project = project
        report.algorithm = algorithm.lower()
        report.all_data = json.dumps(outputs, separators=(',', ':'))
        report.selected_document_index = selected_document_index
        report.selected_document_name = selected_document_name
        report.save()

        return redirect('view_similarity_report', project.id, algorithm, report.id)

    return render(request, 'document_similarity/params.html', content)


def view_similarity_report(request, project_pk, algorithm, report_pk):
    project = get_object_or_404(Project, pk=project_pk)
    report = get_object_or_404(Report, pk=report_pk, algorithm=algorithm.lower())
    files = project.get_files()

    content = {
        'project': project,
        'algorithm': algorithm,
        'files': files,
        'report': report,
        'selected_document_index': report.selected_document_index,
        'outputs': report.get_output()
    }

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Document Similarity": reverse('similarity_algorithms', args=[project_pk]),
        algorithm.upper(): reverse('apply_similarity_algorithm', args=[project_pk, algorithm]),
        f"Report (id:{report.id})": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'document_similarity/report.html', content)


def remove_similarity_report(request, project_pk, algorithm, report_pk):
    report = get_object_or_404(Report, pk=report_pk, project_id=project_pk)
    report.delete()

    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
