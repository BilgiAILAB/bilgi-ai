from django.shortcuts import render, redirect, get_object_or_404

# Create your views here.
from project.models import Project, ProjectFile


def list_projects(request):
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
        print(request.FILES)
        for f in files:
            file_instance = ProjectFile(file=f, project=project)
            file_instance.save()

        return redirect('list_projects')

    return render(request, "project/project-new.html")


def remove_project(request, pk):
    project = get_object_or_404(Project, pk=pk)
    project.delete()

    return redirect('list_projects')
