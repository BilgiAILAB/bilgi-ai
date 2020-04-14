# How to install

1. [Download](https://github.com/ibrahim-dogan/cmpe-graduation-project/archive/master.zip) or clone repository with command-line:
```
git clone https://github.com/ibrahim-dogan/cmpe-graduation-project.git
```
2. Install requirements
```
cd cmpe-graduation-project
pip install -r requirements.txt
```
3. Run this commands to initialize DB file
```
python manage.py migrate
python manage.py collectstatic
```
4. You can run the server now! 
```
python manage.py runserver
```
5. Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to see the website.
