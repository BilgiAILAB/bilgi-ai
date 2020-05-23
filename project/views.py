import os
import shutil
import zipfile

from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse

from project.helpers import pdf_to_text
from project.models import Project, ProjectFile


def all_projects(request):
    projects = Project.objects.all()

    content = {'projects': projects}
    return render(request, "project/project-list.html", content)


def show_project(request, pk):
    project = get_object_or_404(Project, pk=pk)

    content = {'project': project}
    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: "",
    }

    content['breadcrumb'] = breadcrumb
    return render(request, "project/project-show.html", content)


def create_project(request):
    if request.method == 'POST':
        name = request.POST['name']
        project = Project(title=name)
        project.save()

        files = request.FILES.getlist('files')
        for file in files:
            if file.name.endswith('.pdf'):
                file_instance = ProjectFile(project=project)
                file_instance.file_pdf = file  # PDF
                file = pdf_to_text(file)
                file_instance.file = file  # TXT
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
    project_folder = f'data/projects/{project.project_folder}'
    try:
        shutil.rmtree(project_folder)
    except FileNotFoundError:
        pass
    finally:
        pass
    project.delete()

    return redirect('all_projects')


def add_files(request, pk):
    if request.method == 'POST':
        project = get_object_or_404(Project, pk=pk)

        files = request.FILES.getlist('files')
        for file in files:
            if file.name.endswith('.pdf'):
                file_instance = ProjectFile(project=project)
                file_instance.file_pdf = file  # PDF
                file = pdf_to_text(file)
                file_instance.file = file  # TXT
                file_instance.save()
            elif file.name.endswith('.txt'):
                file_instance = ProjectFile(file=file, project=project)
                file_instance.save()
            elif file.name.endswith('.zip'):
                pass

        return redirect('show_project', pk=pk)
    return None


def download(request, pk):
    """Download archive zip file of code snippets"""
    response = HttpResponse(content_type='application/zip')
    zf = zipfile.ZipFile(response, 'w')

    # create the zipfile in memory using writestr
    # add a readme
    # zf.writestr("README_NAME", "README_CONTENT")
    project = get_object_or_404(Project, pk=pk)
    # retrieve snippets from ORM and them to zipfile
    files = ProjectFile.objects.filter(project_id=pk)

    for file in files:
        if file.file_pdf is not None:
            file_path = file.get_project_folder(file.filename_pdf())
            fdir, fname = os.path.split(file_path)
            zip_subdir = "pdfs"
            zip_path = os.path.join(zip_subdir, fname)
            # Add file, at correct path
            zf.write(file_path, zip_path)

        if file.file is not None:
            file_path = file.get_project_folder(file.filename())
            fdir, fname = os.path.split(file_path)
            zip_subdir = "txts"
            zip_path = os.path.join(zip_subdir, fname)
            # Add file, at correct path
            zf.write(file_path, zip_path)

    ZIPFILE_NAME = f"{project.project_folder}.zip"
    # return as zipfile
    response['Content-Disposition'] = f'attachment; filename={ZIPFILE_NAME}'
    return response
