from django.urls import path
from . import views
app_name = 'clasification'




urlpatterns = [
    path('index/',views.index_view,name = 'index_view'),
    path('directions-jahat/',views.directions_jahat,name = 'directions_jahat'),
    path('classificatin-view/',views.classification_view,name = 'classification_view'),
    path('receive-json/', views.receive_json, name='receive_json'),
]