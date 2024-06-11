from django.urls import path
app_name = "check_impedance"
from . import views

urlpatterns = [
    path("check-device/",views.check_device_view,name="check-device"),
    path("config-device/",views.config_device_view,name="config-device"),
    path("check-impedance/",views.check_impedance_view,name="check-impedance"),
    path("get-data/",views.get_data_view,name="get-data"),
    path("save_signal/",views.save_signal_data_view,name="save-signal-data"),
    #path("classification-signal/<int:id>/",views.classification_signal,name="classification-signal"),
]
