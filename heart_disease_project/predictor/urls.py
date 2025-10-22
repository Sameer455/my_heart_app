from django.urls import path
from .views import index, predict,about,diet,workout,history

urlpatterns = [
    path('', index, name='index'),
    path('about/', about, name='about'),  # About page
    path('diet/', diet, name='diet'),  # Diet page
    path('workout/', workout, name='workout'),
    path('history/', history, name='history'),
    path('predict/', predict, name='predict'),
   
]
