from django.shortcuts import render


# Create your views here.
def home(request):
    content = {'title': 'Home'}
    return render(request, "index.html", content)
