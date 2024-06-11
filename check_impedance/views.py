from django.shortcuts import render,HttpResponse
from .sarmad import device,device_settings
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#import os
#import mne

from time import sleep
from .forms import ConfigForms
# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import ConfigSerializers,SaveSignalSerializer
import json




a = signal.butter(3,np.array([4,10]),'bandpass',False,'ba',device_settings.sampling_rate)
bleadoff,aleadoff = a[0],a[1]

device_status = "on"




@api_view(["GET"])
def check_device_view(request):
    global device_status
    
    if device.connect():
        print("device is connected.")
        device_status = "on"
        device_name = device.getInfo('name')
        firmware_version = 'FirmwareV'+ device.getInfo('version').replace("",".")[1:-1]
        return Response({"status":200,"device_status":device_status,"device_name":device_name,"firmware_version":firmware_version})
        
    else :
        return Response({"status":200,"device_status":"please chech again."})
    
    
    
@api_view(['GET', 'POST'])
def config_device_view(request):
    if request.method == 'POST':
        if device_status == "on":
            data = request.data

            ch_count = 21
            ex_count = 3
            

            leadoff_mode = int(data['leadoff_mode'])
            sampling_rate = int(data['sampling_rate'])

            
            if (leadoff_mode == 0):
                device_settings.get_leadoff_mode = False
            else:
                device_settings.get_leadoff_mode = True
        
            device_settings.sampling_rate = sampling_rate
    
            device_settings.gain = 24
            
            device_settings.exgain = 24

            try:
                device.set(device_settings)
                return Response({"status": "200", "data": request.data})
            except:
                
                return Response({"message": "error"})
            
        else:
            return Response({"message": "device is off."})
    else:
        return Response({"message": device_status,"data":"{'leadoff_mode':'','sampling_rate':''}"})
        

    

channel_ampedance_check_main_data = {}
channel_ampedance_check_ext_data = {}   

 
@api_view(['GET'])
def check_impedance_view(request):
    
        #check ampedanse
    channel_ampedance_check_main_data = {}
    channel_ampedance_check_ext_data = {}
    ch_count = 21
    ex_count = 3
    
    def res_calculator(i,main_leadoff_data,ext_leadoff_data): #i is channel index
        
        my_cap = 100 #68nF
        my_res = 15 #30K ohm
        my_current = 24 #nA
        my_freq = 7.8 #Hz
        c_impedance = 1/(2*np.pi*my_freq*my_cap)
        rc_current = my_current * c_impedance
        c_impedance *= 1e9 

        if i>8:
            #9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
            tempedit = main_leadoff_data[i]
        else:
            #6 7 8
            tempedit = ext_leadoff_data[i]


        edited_input = signal.filtfilt(bleadoff,aleadoff,np.array(tempedit))
        dv = (np.sqrt(np.mean(np.array(edited_input)**2)))
        res_val = ((c_impedance * dv) / (rc_current-dv))
        res_val = np.abs(int(res_val/1000)) - my_res
        
        return res_val
    
    def get_data_for_check_ampedanse(record_time=3):
        global channel_ampedance_check_main_data
        global channel_ampedance_check_ext_data
        
        device.start()
        print("****start recording signal****")
        sleep(record_time)
        device.stop()
        print("****stop recording signal\n\n****")
        
        print("processing the record data...")
        r = device.getData()
        
        r_range = range(r.Length)
        
        feedbacks = []
        ext_leadoff_data = {}
        main_leadoff_data = {}
        
        for ind_ext in range(6,ex_count+6):
            ext_leadoff_data[ind_ext]=[]
        
        #make dict for save 21 channel data
        for ind in range(ex_count+6,ex_count+ch_count+6):
            main_leadoff_data[ind]=[]
        
        online_window = 2*device_settings.sampling_rate
        
        for j in r_range:
            f = int(r[j][4])
            feedbacks.append(f)
            for i in range(6,ex_count+6):
                ext_leadoff_data[i].append(1e-6*r[j][i])
                if len(ext_leadoff_data[i]) > online_window:
                    ext_leadoff_data[i].pop(0)
                
            for k in range(ex_count+6,ex_count+ch_count+6):
                main_leadoff_data[k].append(1e-6*r[j][k])
                if len(main_leadoff_data[k])>online_window:
                    main_leadoff_data[k].pop(0)
                    
        
    
        print("**\n\n process Data Done. \n\n**")
        
    ################ampedance check main channel

        print("checking ampedance...\n")
        
        for channel_create_array in range(ex_count+6,ex_count+ch_count+6):
            channel_ampedance_check_main_data[channel_create_array] = []

        ###############ampedance check external channel
        for channel_create_array_ext in range(6,ex_count+6):
            channel_ampedance_check_ext_data[channel_create_array_ext] = []

        
        for check_ampedance_for_each_channel in range(ex_count+6,ex_count+ch_count+6):
            channel_ampedance_check_main_data[check_ampedance_for_each_channel].append(res_calculator(check_ampedance_for_each_channel,main_leadoff_data,ext_leadoff_data))
        

        
        for check_ampedance_for_each_channel_ext in range(6,ex_count+6):
            channel_ampedance_check_ext_data[check_ampedance_for_each_channel_ext].append(res_calculator(check_ampedance_for_each_channel,main_leadoff_data,ext_leadoff_data))
        print("done check ampedance.\n")

        
        #get data for plot or other work
        
        main_data = []
        
        for m in range(ex_count+6,ex_count+ch_count+6):
            main_data.append(main_leadoff_data[m])
        
        main_data = np.array(main_data)

        exe_data = []
        
        for e in range(6,ex_count+6):
            exe_data.append(ext_leadoff_data[e])
        
        exe_data = np.array(exe_data)

        return main_data,exe_data,channel_ampedance_check_main_data,channel_ampedance_check_ext_data

    if device_status=="on":
        RUN =True
        repeat_numper = 2
        count = 0

        record_time = 3

        main_data,exe_data,channel_ampedance_check_main_data,channel_ampedance_check_ext_data=get_data_for_check_ampedanse(record_time)
        return Response({"channel_ampedance_check_main_data":channel_ampedance_check_main_data})
    
    # while (RUN):
    #     main_data,exe_data,channel_ampedance_check_main_data,channel_ampedance_check_ext_data=get_data_for_check_ampedanse(record_time)
        
    #     print("ampedance for main channel = \n",channel_ampedance_check_main_data)
    #     print("ampedance for external channel = \n",channel_ampedance_check_ext_data)
        
    #     count+=1
        
    #     if repeat_numper<count:
    #         RUN=False
    else:
        return Response({"status":200,"device_status":"off"})

                

    
    
    
    # return Response({"device status":{device_status},"sampling_rate":device_settings.sampling_rate})








@api_view(['GET'])
def get_data_view(request):
    
        #check ampedanse
      
    ch_count = 21
    ex_count = 3
    
    
    def get_data(record_time=3):
        
        device.start()
        print("****start recording signal****")
        sleep(record_time)
        device.stop()
        print("****stop recording signal\n\n****")
        
        print("processing the record data...")
        r = device.getData()
        
        r_range = range(r.Length)
        
        feedbacks = []
        ext_data = {}
        main_data = {}

        for ind_ext in range(6,ex_count+6):
            ext_data[ind_ext]=[]
            
        for ind in range(ex_count+6,ex_count+ch_count+6):
            main_data[ind]=[]
            


        for j in r_range:
            f = int(r[j][4])
            feedbacks.append(f)
            
            for i in range(6,ex_count+6):
                ext_data[i].append(0.000001*r[j][i])
            
            
                
            for k in range(ex_count+6,ex_count+ch_count+6):
                main_data[k].append(0.000001*r[j][k])
                    
        
    
        print("**\n\n process Data Done. \n\n**")
        
    ################ampedance check main channel

        #get data for plot or other work
        return main_data,ext_data

    if device_status=="on":
        RUN =True
        repeat_numper = 2
        count = 0



        record_time = 3

        main_data,exe_data=get_data(record_time)
        
        #write data in json file
        with open("main_data.json","w") as output_main_data:
            json.dump(main_data,output_main_data)
        
        
        with open("exe_data.json","w") as output_exe_data:
            json.dump(exe_data,output_exe_data)
        
        return Response({"main_data":main_data,"exe_data":exe_data})
    
    
    
    
    
    # while (RUN):
    #     main_data,exe_data,channel_ampedance_check_main_data,channel_ampedance_check_ext_data=get_data_for_check_ampedanse(record_time)
        
    #     print("ampedance for main channel = \n",channel_ampedance_check_main_data)
    #     print("ampedance for external channel = \n",channel_ampedance_check_ext_data)
        
    #     count+=1
        
    #     if repeat_numper<count:
    #         RUN=False
    else:
        return Response({"status":200,"device_status":"off"})


@api_view(['GET', 'POST'])
def save_signal_data_view(request):
    
    if request.method == 'POST':
        
        data = request.data
         
        first_name = data['firstName']
        last_name = data['lastName'] 
        age = data['age']
        right_left_mix_hand = data['hand']
        gender = data['gender']
        phone = data['phone']
        national_code = data['nationalCode']
        address = data['address']
        
        tag = data['tag']
        sampling_rate = data['sampleRate']
        
        with open("E:\\record_eeg\\eeg_project_web\\main_data.json",'r') as read_main_data:
            
            main_data = json.load(read_main_data)
        
        with open("E:\\record_eeg\\eeg_project_web\\main_data.json",'r') as read_exe_data:
            
        
            exe_data = json.load(read_exe_data)
        
        
        serialize_signal = SaveSignalSerializer(data={
            "tag":tag,
            "sampling_rate":sampling_rate,
            "main_data":main_data,
            "exe_data":exe_data,
            "first_name":first_name,
            "last_name":last_name,
            "age":age,
            "right_left_mix_hand":right_left_mix_hand,
            "gender":gender,
            "phone":phone,
            "national_code":national_code,
            "address":address,})
        
        if serialize_signal.is_valid():
            serialize_signal.save()
            

        return Response({"status":201})