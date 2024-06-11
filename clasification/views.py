from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


def index_view(request):
    return render(request,'clasification/index.html')



def directions_jahat(request):

    return render(request,'clasification/directions_jahat.html')

def classification_view(request):
    with open("E:\\record_eeg\\eeg_project_web\\clasification\\configuration.json","r") as f:
        data = json.load(f)
        
    context = {
        "IP_SERVER":data['IP_SERVER'],
        "IP_CLIENT":data['IP_CLIENT'],
        "POWER_HIGH":data["POWER_HIGH"],
        "POWER_LOW":data["POWER_LOW"]
        }
    return render(request,'clasification/classification.html',context)



@csrf_exempt  # Disable CSRF protection for this view
def receive_json(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Process the data here
            print(data)  # For demonstration purposes
            return JsonResponse({'status': 'success', 'data': data})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)