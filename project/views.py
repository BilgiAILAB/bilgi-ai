from django.shortcuts import render, redirect, get_object_or_404

from project.helpers import pdf_to_text
from project.models import Project, ProjectFile


def all_projects(request):
    projects = Project.objects.all()

    content = {'projects': projects}
    return render(request, "project/project-list.html", content)


def show_project(request, pk):
    project = get_object_or_404(Project, pk=pk)

    content = {'project': project}
    return render(request, "project/project-show.html", content)


def create_project(request):
    if request.method == 'POST':
        name = request.POST['name']
        project = Project(title=name)
        project.save()

        files = request.FILES.getlist('files')
        for file in files:
            if file.name.endswith('.pdf'):
                file = pdf_to_text(file)
                file_instance = ProjectFile(file=file, project=project)
                file_instance.save()
            elif file.name.endswith('.txt'):
                file_instance = ProjectFile(file=file, project=project)
                file_instance.save()
            elif file.name.endswith('.zip'):
                pass

        return redirect('all_projects')

    return render(request, "project/project-new.html")


def delete_project(request, pk):
    project = get_object_or_404(Project, pk=pk)
    project.delete()

    return redirect('all_projects')


def add_files(request, pk):
    if request.method == 'POST':
        project = get_object_or_404(Project, pk=pk)

        files = request.FILES.getlist('files')
        for file in files:
            if file.name.endswith('.pdf'):
                file = pdf_to_text(file)
                file_instance = ProjectFile(file=file, project=project)
                file_instance.save()
            elif file.name.endswith('.txt'):
                file_instance = ProjectFile(file=file, project=project)
                file_instance.save()
            elif file.name.endswith('.zip'):
                pass

        return redirect('show_project', pk=pk)
    return None
