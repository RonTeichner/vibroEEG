
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:18:00 2020

@author: Elad David
"""


import sys
import mne
import matplotlib.pyplot as plt
import time

import scipy.io as sio
import numpy as np
from scipy import signal
import padasip as pa
import array as arr

class MyFilter:
    def __init__(self):
        super().__init__() 
        self.zi_notch=[]
        self.zi_band=[]
        self.welch_1=18
        self.welch_2=24
        self.idx_topo=np.array(range(64))
        self.freq_1_y=20
        self.freq_2_y=20
        self.order = 2
        self.fs = 500
        self.lowcut = 10
        self.highcut = 40
        self.nyq = 0.5 * self.fs
        self.low = self.lowcut / self.nyq
        self.high = self.highcut / self.nyq
        
        self.f0 = 50.0  # Frequency to be removed from signal (Hz)
        self.Q = 30.0  # Quality factor
        #Design notch filter
        
        self.b_notch, self.a_notch = signal.iirnotch(self.f0, self.Q, self.fs)
        self.b_band, self.a_band = signal.butter(self.order, [self.low, self.high], btype='band')
        for i in range(70):
            self.zi_notch.append(signal.lfilter_zi(self.b_notch, self.a_notch))
            self.zi_band.append(signal.lfilter_zi(self.b_band, self.a_band))
        
    def Filter_data(self,data):
        #  FILTERING DATA           
        # =============================================================================
        data, zf_notch = signal.lfilter(self.b_notch, self.a_notch, data, axis=-1, zi=self.zi_notch)
        data, zf_band = signal.lfilter(self.b_band, self.a_band, data, axis=-1, zi=self.zi_band)
        self.zi_notch = zf_notch
        self.zi_band = zf_band
        return data




NumOfChannels = 70
vhdr_file = 'D:\\Downloads\\EEG_Data_03.vhdr'
raw_data = mne.io.read_raw_brainvision(vhdr_file, misc='auto', scale=1e6)
#filtered_data=Filter_data(raw_data)
#ind = np.linspace(1,NumOfChannels,NumOfChannels,dtype=int)
vec = (raw_data.get_data(start=0,stop=5000))
#mat = np.zeros((NumOfChannels,int(vec.size/NumOfChannels)+1))
#list1 = list(range(0,vec.size,NumOfChannels))
#newList = list1
filt = MyFilter()
filtered_data=filt.Filter_data(vec)
#Seperate data to different channels - stored in mat
#for i in range(NumOfChannels):
#    newList = list(range(i,vec.size,NumOfChannels))
#    mat[i][list(range(len(newList)))] = vec[0][newList]
#    mat[i] = filt.Filter_data(mat[i])

#mat = mat[:,:-70]

fig,grph = plt.subplots(2)

grph[0].plot(filtered_data[60])
grph[1].plot(filtered_data[2])
plt.show()



