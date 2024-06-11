from glob import glob
import os
from scipy.io import loadmat
from scipy import signal,stats
import pandas as pd
import numpy as np
from numpy.fft import fft , ifft
import pywt
import matplotlib.pyplot as plt 
import re
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_selection import RFE
import tensorflow as tf
from tensorflow import keras 
from keras.utils import to_categorical
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn import metrics
#import mne

import pickle5 as pickle
import mne
from mne import io
from mne.datasets import sample

#from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,corrmap)


map_label = [
        {'shoroT1':0,'balaT1':1,'paeinT1':2,'raastT1':3,'chapT1':4,'jeloT1':5,'aqabT1':6,'payanT1':7},
        {'shoroT2':0,'balaT2':1,'paeinT2':2,'raastT2':3,'chapT2':4,'jeloT2':5,'aqabT2':6,'payanT2':7},
        {'shoroT3':0,'balaT3':1,'paeinT3':2,'raastT3':3,'chapT3':4,'jeloT3':5,'aqabT3':6,'payanT3':7},          
]


def filter_noisy_data(data,channel_num,sample_for_data,fs,f_nyq,f_low,f_high):
    
    '''
        در این تابع تمام کارهای فیلتر کردن سیگنال انجام می‌شود.
          به منظور پیاده سازی روند فیلترینگ سیگنال، از فیلتر باترورث پیوسته استفاده شده است که معمولا از نوع میان گذر آن استفاده شده است. 
    '''
    
    data_filter = np.zeros_like(data)
    Wn = [f_low,f_high]/np.float_(f_nyq)
    order_filter = 4
    (b,a) = signal.butter(order_filter, Wn, btype='bandpass', output='ba')

    for channel_signal in range(0,channel_num):
        
        '''
         در این قسمت چون سیگنال ما از ۲۱ کانال گرفته شده است هر کدام از کانال ها را فیلتر می‌کنیم.
         سپس یک فور میزنیم به طول تعداد کانال ها.
         بعد دیتای فیلتر شده را درون یک ماتریس صفر که از قبل تعریف کرده ایم می‌ریزیم.
        '''
        
        data_each_channel = data[channel_signal,:]
        data_filter[channel_signal,:] = signal.filtfilt(b,a,data_each_channel)

    return data_filter





#برای استخراج ویژگی آماری از این تابع استفاده می‌کنیم.
def statistical_features(data_for_feature_extraction):
    
    MEAN = np.mean(data_for_feature_extraction)
    VAR = np.var(data_for_feature_extraction)
    ptp = np.ptp(data_for_feature_extraction)
    minim = np.min(data_for_feature_extraction)
    maxim = np.max(data_for_feature_extraction)
    rms = np.sqrt(np.mean(data_for_feature_extraction**2))
    POWER = np.mean(np.power(data_for_feature_extraction,2))
    SKEW = stats.skew(data_for_feature_extraction)
    KUR = stats.kurtosis(data_for_feature_extraction) 
    
    #abs_diff_signal = np.sum(np.abs(np.diff(x)))
    return (MEAN,VAR,ptp,minim,maxim,rms,POWER,SKEW,KUR)






def time_feature_selection(data,channel_num):
    '''
        داخل آرگومان اول تابع دیتای هر ۵ باند وجود دارد که میتوانیم یکی یکی آن را دریافت کنیم بعد فیچرهای هر کانال هر باند را درون یک ماتریس 
        ذخیره کنیم که در نهایت یک تنسور ۵*۲۱*۵ داریم که ۵ اول تعداد باندهاس،۲۱ تعداد کانال، ۵ تعداد فیچرهای استخراجی می‌باشد.
         ماتریس اول داخل تنسور فیچرهای اسخراج شده از باند دلتا برای ۲۱ کانال است که یک ماتریس ۲۱* ۵ میشه که ۵ تعداد فیچرهای استخراجی می‌باشد و غیره       
    '''
    
    num_feature = 9 #mean var power skew kur 
    time_feature_selections_total_bands = np.zeros((channel_num,num_feature))


        
    for i in range(0,channel_num):
        data_for_feature_extraction = data[i,:]
    
        time_feature_selections_total_bands[i,:] = statistical_features(data_for_feature_extraction)
        

    return time_feature_selections_total_bands
    


def seperate_band(data,channel_num,sample_for_data,fs,f_nyq):
    #در حوزه زمان ریتم های سیگنال را استخراج می‌کنیم
    #یک ماتریس مثلا ۲۱*۳۹۰۰ که همش سیگنال های باند دلتا را دارد
    #برای هر کانال باندهای آن را استخراج می‌کنیم.
    delta = filter_noisy_data(data,channel_num,sample_for_data,fs,f_nyq,f_low=1,f_high=4)
    
    #یک ماتریس مثلا ۲۱*۳۹۰۰ که همش سیگنال های باند تتا را دارد
    theta = filter_noisy_data(data,channel_num,sample_for_data,fs,f_nyq,f_low=4,f_high=8)
    
    #یک ماتریس مثلا ۲۱*۳۹۰۰ که همش سیگنال های باند آلفا را دارد     
    alpha = filter_noisy_data(data,channel_num,sample_for_data,fs,f_nyq,f_low=8,f_high=12)

    #یک ماتریس مثلا ۲۱*۳۹۰۰ که همش سیگنال های باند بتا را دارد
    beta = filter_noisy_data(data,channel_num,sample_for_data,fs,f_nyq,f_low=12,f_high=30)

    #یک ماتریس مثلا ۲۱*۳۹۰۰ که سیگنال های باند گاما در هر کانال را دارد
    gamma = filter_noisy_data(data,channel_num,sample_for_data,fs,f_nyq,f_low=30,f_high=40)

    return (delta,theta,alpha,beta,gamma)




#از این تابع برای تبدیل فوریه گرفتن از سیگنال استفاده می‌کنیم.
def frequency_domain(data):
    #در این قسمت می‌خوایم از سیگنال تبدیل فوریه بگیریم و برای هر کانال این کار را می‌کنیم..
    N = data.shape[1]
    channel_num = data.shape[0]
    matrix_for_save_fft_each_channel = np.zeros_like(data)
    
    for i in range(0,channel_num):
        #به ازای هر کانال یک سیگنال یک اف اف تی می‌گیریم.
        fft_data = fft(data[i,:],N)
        matrix_for_save_fft_each_channel[i,:] = fft_data
    #خروجی: تبدیل فوریه هر ۲۱ کانال در یک ماتریس به طول سیگنال ورودی 
    return matrix_for_save_fft_each_channel



def feature_selection_frequency(data):
    
    N = data.shape[1]
    channel_num = data.shape[0]
    fs = 2000
    f_r = np.linspace(0,fs/2,int(np.floor(N/2)))
    f_r_total = np.concatenate((f_r,f_r[::-1]))

    num_feature = 9 #mean var power skew kur 
    
    feature_selections_total = np.zeros((channel_num,num_feature)) #number_f_bands = 5
    
    for ch_num in range(channel_num):
        #اطلاعات باند فرکانسی یک کانال

        bands_range_frequency = np.abs(data[ch_num,:int(np.floor(N/2))])

        feature_selections_total[ch_num,:] = statistical_features(bands_range_frequency)

        #f_r = f_r[f_r<40]
        #plt.stem(f_r,bands_range_frequency[:len(f_r)])
        #plt.show()

    return feature_selections_total

def feature_selectin_wavelet(data,channel_num):
    original_sfreq = 2000
    target_sfreq = 120
    resampling_factor = original_sfreq/target_sfreq
    wavelet = 'db6'
    level = 4
    
    number_of_bands = 5
    number_featuers = 9
    wavelete_feature_extractions = np.zeros((number_of_bands, channel_num, number_featuers))
    
    for i in range(channel_num):
        
        resampled_eeg_data = signal.resample(data[i,:],int(len(data[i,:])/resampling_factor))
        
        #(a4,d4,d3,d2,d1)
        #(delta,theta,alpha,beta,gamma)
        
        bands = pywt.wavedec(resampled_eeg_data, wavelet, mode='symmetric', level=level)
        
        for index,band in enumerate(bands):
            wavelete_feature_extractions[index,i,:] = statistical_features(band)
            
    return wavelete_feature_extractions


        
def read_raw_data(path):
    
    fs = 2000
    f_nyq = fs/2
    
    Temp = loadmat(path)
    a = 7e-7
    data = Temp['EEG_Data']
    
    #determine the label for data
    re_for_get_label=re.findall(r'ID.*event_id',str(Temp['Labels'][0]))[0].split(",")[0]
    re_for_get_label=re_for_get_label.split(':')[1].strip()
    label_for_subject_t = re_for_get_label.replace("'",'')

    #determine channel number eeg signal i
    channel_num = data.shape[0]
    #determine sample number for eeg signal i
    sample_for_data = data.shape[1]
    
    ####################################################denoising data 
    #determine 8 - 30 hz

    data_denoising = filter_noisy_data(data, channel_num, sample_for_data, fs, f_nyq, f_low=8,f_high=30)

    #از این به بعد این دیتای دینویز را برای اسخراج ویژگی به توابع می‌فرستیم
    #####################################################seperate frequency band
    #(delta_time,theta_time,alpha_time,beta_time,gamma_time) = seperate_band(data_denoising,channel_num,sample_for_data,fs,f_nyq)
    
    #####################################################Time Feature Extraction
    time_feature_selection_for_entire_band = time_feature_selection(data_denoising,channel_num)
    
    ############################################################frequency domain + frequency Feature Extraction
    #تبدیل فوریه داده دینویز شده
    change_time_domain_to_frequency_domain_data = frequency_domain(data_denoising)
    #استخراج ویژگی از حوزه فرکانس
    frequency_feature_selection_for_entire_bands = feature_selection_frequency(change_time_domain_to_frequency_domain_data)
    
    #استخراج ویژگی از حوزه ویولت
    data_denoising_for_wavelet = filter_noisy_data(data, channel_num, sample_for_data, fs, f_nyq, f_low=1,f_high=40)
    wavelet_feature_selection_for_entire_bands = feature_selectin_wavelet(data_denoising_for_wavelet,channel_num)
    
    return label_for_subject_t, time_feature_selection_for_entire_band,frequency_feature_selection_for_entire_bands,wavelet_feature_selection_for_entire_bands


path_os = os.getcwd()
PATH = os.path.join(path_os,'data','v1')

# map_label = [
#         {'shoroT1':1,'balaT1':2,'paeinT1':3,'raastT1':4,'chapT1':5,'jeloT1':6,'aqabT1':7,'payanT1':8},
#         {'shoroT2':1,'balaT2':2,'paeinT2':3,'raastT2':4,'chapT2':5,'jeloT2':6,'aqabT2':7,'payanT2':8},
#         {'shoroT3':1,'balaT3':2,'paeinT3':3,'raastT3':4,'chapT3':5,'jeloT3':6,'aqabT3':7,'payanT3':8},          
# ]

w6 = []
w7 = []
w8 = []
w9 = []
w10 = []
w11 = []
w15 = []
w16 = []


data_for_network = []
#data_for_network2 = np.zeros((len(subject_T),chan_numbers*num_fetures))
label_y = []

for subject in range(1,7):
    subject_T = glob(PATH+f'/s{subject}/Directions_and_Time_T{1}/*.mat')
    Path_for_save = PATH+f'/s{subject}/Directions_and_Time_T{1}/FetureExtraction/'

    #num_fetures = 9
    #total_features = 25
    #chan_numbers = 21 #channel numbers
    #bands_numbers = 5
    

    
    for index,data_raw_path in enumerate(subject_T):
          
        if ("W6" in data_raw_path):
            w6.append(data_raw_path)
        elif ("W7" in data_raw_path):
            w7.append(data_raw_path)
        elif ("W8" in data_raw_path):
            w8.append(data_raw_path)
        elif ("W9" in data_raw_path):
            w9.append(data_raw_path)
        elif ("W10" in data_raw_path):
            w10.append(data_raw_path)
        elif ("W11" in data_raw_path):
            w11.append(data_raw_path)
        elif ("W15" in data_raw_path):
            w15.append(data_raw_path)
        elif ("W16" in data_raw_path):
            w16.append(data_raw_path)

# for subject in range(7,13):
#     path = f"/home/nika/Desktop/HodHodDataSet/Subject{subject}/Directions/*.mat"  
#     subject_T = glob(path)
#     for index,data_raw_path in enumerate(subject_T):
#         if (("W6" in data_raw_path) and ("T1" in data_raw_path)):
#             w6.append(data_raw_path)
#         elif (("W7" in data_raw_path) and ("T1" in data_raw_path)):
#             w7.append(data_raw_path)
#         elif (("W8" in data_raw_path) and ("T1" in data_raw_path)):
#             w8.append(data_raw_path)
#         elif (("W9" in data_raw_path) and ("T1" in data_raw_path)):
#             w9.append(data_raw_path)
            




#subject_T = glob(PATH+f'/s{subject}/Directions_and_Time_T{1}/*.mat')
#Path_for_save = PATH+f'/s{subject}/Directions_and_Time_T{1}/FetureExtraction/'

num_fetures = 9
total_features = 25
chan_numbers = 21 #channel numbers
bands_numbers = 5

label_y_6 = []
label_y_7 = []
label_y_8 = []
label_y_9 = []
label_y_10 = []
label_y_11 = []

data_for_t = []
data_for_f = []
data_for_w = []

balaT1 = []  
balaF1 = []
balaW1 = []

paeinT1 = []
paeinF1 = []
paeinW1 = []

raastT1 = []
raastF1 = []
raastW1 = []

chapT1 = []
chapF1 = []
chapW1 = []

jeloT1 = []
jeloF1 = []
jeloW1 = []

aqabT1 = []
aqabF1 = []
aqabW1 = []

# jeloT1 = []
# aqabT1 = []
# shoroT1 = []
# payanT1 = []


for index,data_raw_path6 in enumerate(w6):
      

    label6,TFE6,FFE6,WFE6 = read_raw_data(data_raw_path6)

    '''
        فیچرهای که استخراج کردیم به صورت یک تنسور می‌باشد که هر ماتریس آن اطلاعات یک باند فرکانسی است مثلا ماتریس اول شامل ۲۱ سطر است
        و ۵ ستون که به معنی اطلاعات اسخراج شده ۲۱ کانال است
        الان میخوایم اطلاعات ۵ باند را برای هر کانال پشت سر هم قرار بدهیم..
        
        TFE = Time Feature Extraction
        FFE = Frequency Feature Extraction
        WFE = wavelet Feature Extraction

    '''
    
    if ("T1" in label6):
        label_y_6.append(map_label[0].get(label6))

    # data_for_network1[index,:] = np.ravel(TFE)
    # data_for_network2[index,:] = np.ravel(FFE)
    #concat = np.concatenate((data_for_network1,data_for_network2),axis=1)

    WFE_reshape = np.zeros((5*21,9))
    lb = 0
    hb =21
    
    for band_index in WFE6:
        WFE_reshape[lb:hb,:] = band_index
        lb+=21
        hb+=21
    
    

    balaT1.append(np.ravel(TFE6))
    balaF1.append(np.ravel(FFE6))
    balaW1.append(np.ravel(WFE_reshape))
    
# np.save(Path_for_save+f"TFE-{subject}",data_for_network1)
# np.save(Path_for_save+f"FFE-{subject}",data_for_network2)
# np.save(Path_for_save+f"label-{subject}",np.array(label_y))
    # np.save(Path_for_save+"/WFE/"+f"{index}_"+name+"-WFE-"+label,concatenate_WFE_for_each_channel)

for index,data_raw_path7 in enumerate(w7):
      

    label7,TFE7,FFE7,WFE7 = read_raw_data(data_raw_path7)

    '''
        فیچرهای که استخراج کردیم به صورت یک تنسور می‌باشد که هر ماتریس آن اطلاعات یک باند فرکانسی است مثلا ماتریس اول شامل ۲۱ سطر است
        و ۵ ستون که به معنی اطلاعات اسخراج شده ۲۱ کانال است
        الان میخوایم اطلاعات ۵ باند را برای هر کانال پشت سر هم قرار بدهیم..
        
        TFE = Time Feature Extraction
        FFE = Frequency Feature Extraction
        WFE = wavelet Feature Extraction

    '''
    
    if ("T1" in label7):
        label_y_7.append(map_label[0].get(label7))

    # data_for_network1[index,:] = np.ravel(TFE)
    # data_for_network2[index,:] = np.ravel(FFE)
    #concat = np.concatenate((data_for_network1,data_for_network2),axis=1)
    
    WFE_reshape = np.zeros((5*21,9))
    lb = 0
    hb =21
    
    for band_index in WFE7:
        WFE_reshape[lb:hb,:] = band_index
        lb+=21
        hb+=21
    
    

    paeinT1.append(np.ravel(TFE7))
    paeinF1.append(np.ravel(FFE7))
    paeinW1.append(np.ravel(WFE_reshape))



for index,data_raw_path8 in enumerate(w8):
      

    label8,TFE8,FFE8,WFE8 = read_raw_data(data_raw_path8)

    '''
        فیچرهای که استخراج کردیم به صورت یک تنسور می‌باشد که هر ماتریس آن اطلاعات یک باند فرکانسی است مثلا ماتریس اول شامل ۲۱ سطر است
        و ۵ ستون که به معنی اطلاعات اسخراج شده ۲۱ کانال است
        الان میخوایم اطلاعات ۵ باند را برای هر کانال پشت سر هم قرار بدهیم..
        
        TFE = Time Feature Extraction
        FFE = Frequency Feature Extraction
        WFE = wavelet Feature Extraction

    '''
    
    if ("T1" in label8):
        label_y_8.append(map_label[0].get(label8))

    # data_for_network1[index,:] = np.ravel(TFE)
    # data_for_network2[index,:] = np.ravel(FFE)
    #concat = np.concatenate((data_for_network1,data_for_network2),axis=1)
    
    WFE_reshape = np.zeros((5*21,9))
    lb = 0
    hb =21
    
    for band_index in WFE8:
        WFE_reshape[lb:hb,:] = band_index
        lb+=21
        hb+=21
    
    

    raastT1.append(np.ravel(TFE8))
    raastF1.append(np.ravel(FFE8))
    raastW1.append(np.ravel(WFE_reshape))


for index,data_raw_path9 in enumerate(w9):
      

    label9,TFE9,FFE9,WFE9 = read_raw_data(data_raw_path9)

    '''
        فیچرهای که استخراج کردیم به صورت یک تنسور می‌باشد که هر ماتریس آن اطلاعات یک باند فرکانسی است مثلا ماتریس اول شامل ۲۱ سطر است
        و ۵ ستون که به معنی اطلاعات اسخراج شده ۲۱ کانال است
        الان میخوایم اطلاعات ۵ باند را برای هر کانال پشت سر هم قرار بدهیم..
        
        TFE = Time Feature Extraction
        FFE = Frequency Feature Extraction
        WFE = wavelet Feature Extraction

    '''
    
    if ("T1" in label9):
        label_y_9.append(map_label[0].get(label9))

    # data_for_network1[index,:] = np.ravel(TFE)
    # data_for_network2[index,:] = np.ravel(FFE)
    #concat = np.concatenate((data_for_network1,data_for_network2),axis=1)
    
    WFE_reshape = np.zeros((5*21,9))
    lb = 0
    hb =21
    
    for band_index in WFE9:
        WFE_reshape[lb:hb,:] = band_index
        lb+=21
        hb+=21
    
    

    chapT1.append(np.ravel(TFE9))
    chapF1.append(np.ravel(FFE9))
    chapW1.append(np.ravel(WFE_reshape))



class production:
    def __init__(self,name,age,signal):
        self.name = name
        
        self.age = age
        
        self.signal = signal
    
    def show_signal(self):
        return self.signal
    
    def mean_signal(self):
        data = np.mean(self.signal)
        return data
    def var_signal(self):
        data = np.var(self.signal)
        return data
    def psd_signal(self):
        data = np.pad(self.signal)
        return data
    



    
            
        
        
        
        
# for index,data_raw_path in enumerate(w10):
      

#     label,TFE,FFE,WFE = read_raw_data(data_raw_path)

#     '''
#         فیچرهای که استخراج کردیم به صورت یک تنسور می‌باشد که هر ماتریس آن اطلاعات یک باند فرکانسی است مثلا ماتریس اول شامل ۲۱ سطر است
#         و ۵ ستون که به معنی اطلاعات اسخراج شده ۲۱ کانال است
#         الان میخوایم اطلاعات ۵ باند را برای هر کانال پشت سر هم قرار بدهیم..
        
#         TFE = Time Feature Extraction
#         FFE = Frequency Feature Extraction
#         WFE = wavelet Feature Extraction

#     '''
    
#     if ("T1" in label):
#         label_y_9.append(map_label[0].get(label))

#     # data_for_network1[index,:] = np.ravel(TFE)
#     # data_for_network2[index,:] = np.ravel(FFE)
#     #concat = np.concatenate((data_for_network1,data_for_network2),axis=1)
    
#     WFE_reshape = np.zeros((5*21,9))
#     lb = 0
#     hb =21
    
#     for band_index in WFE:
#         WFE_reshape[lb:hb,:] = band_index
#         lb+=21
#         hb+=21
    
    

#     jeloT1.append(np.ravel(TFE))
#     jeloF1.append(np.ravel(FFE))
#     jeloW1.append(WFE_reshape)





# for index,data_raw_path in enumerate(w11):
      

#     label,TFE,FFE,WFE = read_raw_data(data_raw_path)

#     '''
#         فیچرهای که استخراج کردیم به صورت یک تنسور می‌باشد که هر ماتریس آن اطلاعات یک باند فرکانسی است مثلا ماتریس اول شامل ۲۱ سطر است
#         و ۵ ستون که به معنی اطلاعات اسخراج شده ۲۱ کانال است
#         الان میخوایم اطلاعات ۵ باند را برای هر کانال پشت سر هم قرار بدهیم..
        
#         TFE = Time Feature Extraction
#         FFE = Frequency Feature Extraction
#         WFE = wavelet Feature Extraction

#     '''
    
#     if ("T1" in label):
#         label_y_9.append(map_label[0].get(label))

#     # data_for_network1[index,:] = np.ravel(TFE)
#     # data_for_network2[index,:] = np.ravel(FFE)
#     #concat = np.concatenate((data_for_network1,data_for_network2),axis=1)
    
#     WFE_reshape = np.zeros((5*21,9))
#     lb = 0
#     hb =21
    
#     for band_index in WFE:
#         WFE_reshape[lb:hb,:] = band_index
#         lb+=21
#         hb+=21
    
    

#     aqabT1.append(np.ravel(TFE))
#     aqabF1.append(np.ravel(FFE))
#     aqabW1.append(WFE_reshape)


balaT1 = np.array(balaT1[:119])
balaF1 = np.array(balaF1[:119])
balaW1 = np.array(balaW1[:119])

print(balaW1.shape)

paeinT1 = np.array(paeinT1[:119])
paeinF1 = np.array(paeinF1[:119])
paeinW1 = np.array(paeinW1[:119])

print(paeinW1.shape)

raastT1 = np.array(raastT1[:119])
raastF1 = np.array(raastF1[:119])
raastW1 = np.array(raastW1[:119])
print(raastW1.shape)

chapT1 = np.array(chapT1[:119])
chapF1 = np.array(chapF1[:119])
chapW1 = np.array(chapW1[:119])
print(chapW1.shape)

total_data_t = []
total_data_f = []
total_data_w = []
Y = []
Y_f = []

for i,j,w,p, in zip(balaT1,paeinT1,raastT1,chapT1):
    total_data_t.append(i)
    Y.append(6)
    
    total_data_t.append(j)
    Y.append(7)
    
    total_data_t.append(w)
    Y.append(8)

    total_data_t.append(p)
    Y.append(9)

    # total_data_t.append(j)
    # Y.append(10)

    # total_data_t.append(a)
    # Y.append(11)
    
total_data_t = np.array(total_data_t)

for i2,j2,w2,p2 in zip(balaW1,paeinW1,raastW1,chapW1):
    total_data_f.append(i2)
    Y_f.append(6)
    total_data_f.append(j2)
    Y_f.append(7)
    total_data_f.append(w2)
    Y_f.append(8)
    total_data_f.append(p2)
    Y_f.append(9)
    
    # total_data_f.append(j2)
    # Y_f.append(10)

    # total_data_f.append(a2)
    # Y_f.append(11)
    # # total_data_f.append(t2)

total_data_t = np.array(total_data_t)
total_data_f = np.array(total_data_f)

Y = np.array(Y)
Y_f = np.array(Y_f)
Y = Y.reshape(Y.shape[0],1)
Y_f = Y_f.reshape(Y_f.shape[0],1)


#MULTI class
X_train,X_test,y_train,y_test = train_test_split(total_data_t,Y,test_size=0.25,random_state=120)

#
ovr_classifier = OneVsRestClassifier(SVC(kernel='linear' ,C=1.0, decision_function_shape='ovr',degree=3))

# Train the OvR classifier
ovr_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ovr = ovr_classifier.predict(X_test)

# Evaluate the OvR classifier
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
classification_report_ovr = classification_report(y_test, y_pred_ovr)

print("OvR SVM Classifier:")
print(f"Accuracy: {accuracy_ovr}")
print("Classification Report:\n", classification_report_ovr)

confusion_matrix = metrics.confusion_matrix(y_test.transpose().reshape(y_test.shape[0],1),y_pred_ovr)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
plt.show()
plt.figure(0)
# Saving the figure.


