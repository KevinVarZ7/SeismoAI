#!/usr/bin/env python
# coding: utf-8

# <center>
#    <img src="logounam.png"width="150">
# </center>
# 
# 
#   <font size="6"><b> <center> Universidad Nacional Autónoma de México     </b> <br> </font>
#   <font size="4"><b> <center> Posgrado en Ciencias de la Tierra </b><br> </font>
#   <font size="3"> <center> 1D CNN implementation for Regional Seismic Event detection in Paricutin Data </b> <br> </font>
#   <font size="3"><b> <center>@Author: MSc. Kevin Axel Vargas-Zamudio </b> <br> </font>
#   <font size="3"><b> <center>email: seismo.ai.kevvargas@gmail.com </b><br></font>

# # Observed data for 2022 Paricutin Seismic Swarm
# ## 1D CNN development for Regional Seismic Event Detection
# ### Convolutional Neural Network: Generalization test in single station data

# In[3]:


# Base libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy.io import loadmat,savemat
from scipy.signal import decimate
import scipy.io as scio
import matlab.engine

from os.path import dirname, join as pjoin
import sys
import copy as copy
import os
import time
import random


# In[27]:


# Initialization of matlab engine for signal retrieving 
#def connect_2_matlab():
import matlab.engine
    
    #global engine_mat
names = matlab.engine.find_matlab()
    
engine_mat = matlab.engine.connect_matlab(names[0])
    
    #return #engine_mat


# In[5]:


def matfiles_retrieve_24(year,month,day):
    # Retrieving .mat files from /res directory for a single day
    path = sys.path[-1]

    mat_fname = []

    for i in range(0,24):
        #month_day = month + 
        if i <= 9:
            mat_fname.append(path + 'pickreg'+ year + '_'+ month + '_' + day +'_0' + str(i) + '.mat')
        else:
            mat_fname.append(path + 'pickreg'+year+'_'+ month + '_' + day + '_' + str(i) + '.mat')

    # .mat file list for a complete day (each .mat data file for 1 of 24 hours)
    mat_files = []
    for i in range(len(mat_fname)):
        mat_files.append(loadmat(mat_fname[i]))
        
    # Keys for data day dictionary
    data_day = list()
    for s in range(len(mat_fname[:])):
        data_day.append(mat_fname[s][75:-4])
        
    # Create a Dictionary that contains tp and ts variable length vectors for each hour
    # Sorted and repetitions removed

    TPTS_1day = {}
    hours = []
    for i in range(24):
        TP = mat_files[i]['tp'][0] ; TP = TP[~np.isnan(TP)] 
        TS = mat_files[i]['ts'][0] ; TS = TS[~np.isnan(TS)]
        TPTS_1day[data_day[i]+'_tp'] = np.unique(np.sort(TP))
        TPTS_1day[data_day[i]+'_ts'] = np.unique(np.sort(TS))

#         if any(TPTS_1day[data_day[i]+'_tp']) == False or \
#         any(TPTS_1day[data_day[i]+'_ts']) == False:          # Removing hours wihout events
#             TPTS_1day.pop(data_day[i]+'_tp')
#             TPTS_1day.pop(data_day[i]+'_ts')
    
    keys = list(TPTS_1day.keys()) #'2022_09_21_03_tp'
    
    for j in range(0,len(keys),2):
        hours.append(int(keys[j][11:13]))

    return data_day, TPTS_1day, hours


# In[6]:


def matfiles_retrieve_some(year,month,day,hours):
    # Retrieving .mat files from /res directory for a single day
    path = sys.path[-1]

    mat_fname = []

    for h in hours:
        #month_day = month + 
        mat_fname.append(path + 'pick'+year+'_'+ month + '_' + day + '_' + h + '.mat')

    # .mat file list for a complete day (each .mat data file for 1 of 24 hours)
    mat_files = []
    for i in range(len(mat_fname)):
        mat_files.append(loadmat(mat_fname[i]))
        
    # Keys for data day dictionary
    data_day = list()
    for s in range(len(mat_fname[:])):
        data_day.append(mat_fname[s][72:-4])
        #print(data_day[s])
        
    # Create a Dictionary that contains tp and ts variable length vectors for each hour
    # Sorted and repetitions removed

    TPTS_1day = {}
    hours = []
    for i in range(24):
        TP = mat_files[i]['tp'][0] ; TP = TP[~np.isnan(TP)] 
        TS = mat_files[i]['ts'][0] ; TS = TS[~np.isnan(TS)]
        TPTS_1day[data_day[i]+'_tp'] = np.unique(np.sort(TP))
        TPTS_1day[data_day[i]+'_ts'] = np.unique(np.sort(TS))

        if any(TPTS_1day[data_day[i]+'_tp']) == False or         any(TPTS_1day[data_day[i]+'_ts']) == False:          # Removing hours wihout events
            TPTS_1day.pop(data_day[i]+'_tp')
            TPTS_1day.pop(data_day[i]+'_ts')
    
#     keys = list(TPTS_1day.keys()) #'2022_09_21_03_tp'
    
#     for j in range(0,len(keys),2):
#         hours.append(int(keys[j][11:13]))

    return data_day, TPTS_1day


# In[7]:


def visualize_TPTS(hours,TPTS_day):
    
    for h in hours:
        print(f'Hour tp: {h} ',TPTS_day[data_day[h] + '_tp'])
        print(f'Hour ts: {h} ',TPTS_day[data_day[h] + '_ts'])


# In[8]:


def count_local_events(index_array,TPTS_day):
    # Counting how many events respect to tp parameter
    aux = 0
    count_event = 0
    for i in index_array:
        tp = TPTS_day[data_day[i]+ '_tp']
        aux = len(tp)
        count_event += aux
    return count_event


# In[9]:


def time_windowing(Ndec,Nsamp_win,Nover,overlap,index_array):    
    # Time vector and window parameters
    time = np.arange(0,Ndec,1)
    t_end = time[-1]
    t_ini = time[0]

    Nwin = int(np.round((t_end/wind_time)))
    Nhours = len(index_array)
    Nwin_no_overlap = int(Ndec/Nsamp_win)
    Nwin_overlap = int(Nwin_no_overlap/overlap)
    # Windowing 1 signal and overlapping it 

    #print(t_ini,t_end,dt,len(time),Nsamp_w)
#     print(f'Samples/window: {Nsamp_win} \t Num of windows without overlapping: {Nwin_no_overlap}')
#     print(f'Samples Overlapped: {Nover} considering {overlap} of overlap')
#     print(f'Num of windows with overlap: {Nwin_overlap}')
#     print(f'Number of efective detection hours: {Nhours}')
    
    return time, Nhours


# In[10]:


def retrieve_raw_signal(year,month,day,data_day,index_array):
    #print(data_day,index_array)
    import matlab
    
    signal_raw = {}
    day_py   = matlab.double(int(day))
    
    # For interface this number must be between 1 and 6, that are the indices of an array specific for year 2022
    if month == '08':
        month_py = matlab.double(int(2))
    elif month == '09':
        month_py = matlab.double(int(3)) # corresponds to september
    elif month == '10':
        month_py = matlab.double(int(4))
    elif month == '11':
        month_py = matlab.double(int(5))
    elif month == '12':
        month_py = matlab.double(int(6))
        
    # for 2023:
    if month == '01':
        month_py = matlab.double(int(1))
    elif month == '02':
        month_py = matlab.double(int(2))

    if year == '2022':
        year_py = matlab.double(int(4))
    elif year == '2023':
        year_py = matlab.double(int(5))
    elif year == '2021':
        year_py = matlab.double(int(3))
    elif year == '2020':
        year_py = matlab.double(int(2))

    for h in index_array: # loop for selected hours
        signal_raw[data_day[h]+'_sig0'] = np.array(engine_mat.Interface_from_python(year_py,month_py,day_py,h+1))[0,:,:]    

    return signal_raw


# In[11]:


def NRG_signal_decimated(signal_raw,N,k_samp,index_array,fsamp,fsamp_local):
    
    Nr = N - int(10*fsamp) 
    Ndec = int(Nr/k_samp)
    NRG = np.zeros((Nr,24))
    NRG_dec = np.zeros((Ndec,24))
    print(Nr,Ndec)

    for i in (index_array):
        #print(i)
        NRG[:,i] = signal_raw[data_day[i]+'_sig0'][500:359500,3]
        #NRG[:,i] /= np.max(NRG[:,i])
        NRG_dec[:,i] = decimate(NRG[:,i],k_samp)
    
    return NRG_dec, Ndec


# In[12]:


def month_day_str():
    month = []
    day1 = [] ; day2 = []

    for i in range(1,13):
        if i <= 9:
            month.append('0'+str(i))
        else:
            month.append(str(i))
    for i in range(1,31):
        if i <= 9:
            day1.append('0'+str(i))
        else:
            day1.append(str(i))
    day2 = copy.copy(day1)
    day2.append('31')
    
    return month, day1, day2


# In[13]:


def retrieve_noise_hours(mat_files,day_noise):
    
    Noise_matfiles = []
    hours_noise = {}
    
    for j in range(len(mat_files)):
        Noise_matfiles.append(loadmat('X_input_mat/' + mat_files[j]))  
        keys = list(Noise_matfiles[j].keys())
        hours = []
        for i in range(3,len(keys),2):
            #print(keys[i][0:2])
            hours.append(int(keys[i][0:2]))
        
        hours_noise[day_noise[j]] = hours
            
    return hours_noise , Noise_matfiles


# In[14]:


os.path.basename("/StorageCitlalli/Paricutin/EnjambreParicutin/Script_sismo_id_h2/res/")
sys.path.append('/StorageCitlalli/Paricutin/EnjambreParicutin/Script_sismo_id_h2/res/')


# In[15]:


# Sampling and resampling parameters
fsamp_orig = 100 #[Hz]
fsamp_local = 20 #[Hz]
k_samp = int(fsamp_orig / fsamp_local)
# Overlapping window parameters
dt = 1/(100/k_samp)   # Fdr decimated signal to 50 Hz == time * 50
wind_time = 37.5   # s
Nsamp_win = int(wind_time/dt)              # Samples number in each window
overlap = 0.5
Nover = int(np.round((Nsamp_win*overlap)))
print(Nsamp_win,Nover)


# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Model,load_model
# Loading CNN model
model_cnn1d = load_model('CNNModels/CNN1D_3CPFrFDO')
model_cnn1d.summary()


# In[44]:


""" Main program"""
import time

#connect_2_matlab()

months , day_30, day_31 = month_day_str()
days = []
year = '2023'
 
# testing stuff
monthsss = ['01']
daysssss = ['01']
#daysssss = ['01','02','03','04','05','06']
#daysssss = ['07','08','09','10','11','12']
#daysssss = ['13','14','15','16','17','18','19']
#daysssss = ['20','21','22','23','24','25']
#daysssss = ['26','27','28','29','30']
#daysssss = ['06','07','08','09','10','11','12'] 
count_total = 0

index_array = [i for i in range(24)]
#print(index_array)
#start_time = time.time()   # Timing process

count_event = 0

for month in monthsss:
    if  month == '01' or month == '03' or month == '05' or month == '07'     or month == '08' or month == '10' or month == '12':
        days = copy.copy(day_31)
    elif month == '04' or month == '06' or month == '09' or month == '11':
        days = copy.copy(day_30)
  
    for day in daysssss:#days[:]:
        data_day , TPTS_day, hours = matfiles_retrieve_24(year,month,day)
        
        #visualize_TPTS(hours,TPTS_day)
        data_month = data_day[0][0:10]

        """Calling SeisPick program in order to retrieve specific hours with false detections
        Testing calls from here to Matlab seismicpick program interface through function
        Retrieving Raw Signal"""
        
        #print(index_array)
        print(f'\n---------Into Signal Retrieval... Loading Signals {data_month}...-----------\n')
        signal_raw = retrieve_raw_signal(year,month,day,data_day,index_array)

        """ Structures for component retrieve: Signal energy + normalization + decimate """
        if index_array[0] <= 9: 
            N = len(signal_raw[data_month + '_0' + str(index_array[0]) + '_sig0'])
        else:
            N = len(signal_raw[data_month + '_' + str(index_array[0]) + '_sig0'])
            
        NRG_dec, Ndec = NRG_signal_decimated(signal_raw,N,k_samp,index_array,fsamp_orig,fsamp_local)
        time,nhours = time_windowing(Ndec,Nsamp_win,Nover,overlap,index_array) 
        
#         print(NRG_dec[:,index_array[-1]],Ndec,NRG_dec.shape)
#         plt.figure()
#         plt.plot(time[:]/fsamp_local,NRG_dec[:,index_array[-1]],color='red')
        
        nwin_sig = int(np.floor(Ndec/Nsamp_win))-1
        wind_sig = np.zeros((nwin_sig,Nsamp_win,1))
        time_win = np.zeros((nwin_sig,Nsamp_win))
        
        for h in index_array:
            for i in range(nwin_sig):
                wind_sig[i,:,0] = NRG_dec[i*Nsamp_win:Nsamp_win*(i+1),h]/np.max(NRG_dec[i*Nsamp_win:Nsamp_win*(i+1),h])
                time_win[i,:] = time[i*Nsamp_win:Nsamp_win*(i+1)]
                
            wind_sig.reshape(wind_sig.shape[0],wind_sig.shape[1],-1)
            #print(wind_sig.shape)

            proba_class = model_cnn1d.predict(wind_sig)
            prediction_class = np.zeros((len(proba_class[:])),dtype = int)
            
            ev_wind_idx = []
            for k in range(len(proba_class[:])):
                if proba_class[k]<=0.5:
                    prediction_class[k] = 0
                else:
                    prediction_class[k] = 1

                #print(k,proba_class[k],prediction_class[k])
                
                if prediction_class[k] == 1:
                    count_event += 1
                    ev_wind_idx.append(k)
            
            print(f'Count Event for {year}/{month}/{day}, Hour {h}: {count_event}')
            print(ev_wind_idx)
            
#             for e in ev_wind_idx:
#                 plt.figure(figsize=(6,3))
#                 plt.plot(time_win[e,:]/fsamp_local,wind_sig[e,:,0],color='violet')
#                 plt.title(f'Event Detected: {data_month}_{h}, #Window event: {e}')
#                 plt.xlabel('Time [s]')

# #         Counting total events from september to november!
#         count_total += count_event
#         print(f'-------------Total events to {data_month}: {count_total}------------------')


# In[ ]:


Ndec = 72000
nwin_sig = int(np.floor(Ndec/Nsamp_win))-1

for i in range(nwin_sig):
    #wind_sig[:] = NRG_dec[(5*fsamp_local)+i*Nsamp_win:Nsamp_win*(i+1)]
    print(i,(5*fsamp_local)+i*Nsamp_win,Nsamp_win*(i+1)+(5*fsamp_local))


# In[ ]:


Xtest_out = Xtest_out.reshape(Xtest_out.shape[0],Xtest_out.shape[1],-1)


# In[ ]:


p_test_out = model.predict(Xtest_out) #.argmax(axis=0)
p_good_out = np.zeros((len(p_test_out[:])),dtype = int)

for i in range(len(p_test_out[:])):
    if p_test_out[i]<=0.5:
        p_good_out[i] = 0
    else:
        p_good_out[i] = 1

