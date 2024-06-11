from  sarmad import device,device_settings
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import mne
from time import sleep

if device.connect():
    print("device is on")
    device_name = device.getInfo('name')
    firmware_version = 'FirmwareV'+ device.getInfo('version').replace("",".")[1:-1]
    print("device name is = ",device_name)
    print("firmware_version = ", firmware_version)
    if device_name == "Fascin8":
        ch_count = 21
        ex_count = 3
    
    leadoff_mode = int(input("leadoff_mode OFF or ON?(OFF = 0 & ON = 1)"))
    sampling_rate = int(input("Enter sampling rate (250,500,1000,2000): "))
    
    if (leadoff_mode == 0):
        device_settings.get_leadoff_mode = False
    else:
        device_settings.get_leadoff_mode = True
        
    device_settings.sampling_rate = sampling_rate
    
    device_settings.gain = 24
    
    device_settings.exgain = 24
    
    set_settings = int(input("do you want set change settings (NO = 0 or YES=1)? "))
    
    if (set_settings == 1):
        device.set(device_settings)
        
    #check ampedanse
    channel_ampedance_check_main_data = {}
    channel_ampedance_check_ext_data = {}

    a = signal.butter(3,np.array([4,10]),'bandpass',False,'ba',device_settings.sampling_rate)
    bleadoff,aleadoff = a[0],a[1]

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





    def get_data_for_check_ampedanse(record_time):
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


    RUN =True
    repeat_numper = 100
    count = 0
    channel_ampedance_check_main_data = {}
    channel_ampedance_check_ext_data = {}

    for channel_create_array in range(ex_count+6,ex_count+ch_count+6):
        channel_ampedance_check_main_data[channel_create_array] = []

    ###############ampedance check external channel
    for channel_create_array_ext in range(6,ex_count+6):
        channel_ampedance_check_ext_data[channel_create_array_ext] = []

    record_time = int(input("Enter the record time:(1 sec, 3 sec, 5 sec, ...)"))

    while (RUN):
        main_data,exe_data,channel_ampedance_check_main_data,channel_ampedance_check_ext_data=get_data_for_check_ampedanse(record_time)
        
        print("ampedance for main channel = \n",channel_ampedance_check_main_data)
        print("ampedance for external channel = \n",channel_ampedance_check_ext_data)
        
        count+=1
        
        if repeat_numper<count:
            RUN=False
        

        
        





else:
    print("device is off")
    