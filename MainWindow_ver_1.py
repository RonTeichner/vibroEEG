# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:47:42 2020

@author: Igor D
"""

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QRadioButton, QComboBox, QTabWidget , QLineEdit, QSlider
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5 import QtCore
import pyqtgraph as pg
import mne
import matplotlib.pyplot as plt
import time

import scipy.io as sio
import numpy as np
from scipy import signal
import padasip as pa





from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure





class Window(QtGui.QWidget):
    def __init__(self):
        super().__init__()
        
        self.title = "Vibro Tactile Experiment"
        self.top=100
        self.left=100
        self.width=700
        self.height=900
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top,self.width,self.height)
        self.elec_1=7
        self.elec_2=24
        self.welch_1=18
        self.welch_2=24
        self.idx_topo=np.array(range(64))
        self.freq_1_y=20
        self.freq_2_y=20
        
        
        #%% Data reading, Filter and topomap
        
        # Filter parameters
        
        #FILTER PARAMETERS       
     
     
        self.order = 2
        self.fs = 500
        self.lowcut = 10
        self.highcut = 40
        self.nyq = 0.5 * self.fs
        self.low = self.lowcut / self.nyq
        self.high = self.highcut / self.nyq

        self.f0 = 50.0  # Frequency to be removed from signal (Hz)
        self.Q = 30.0  # Quality factor
        # Design notch filter
        self.b_notch, self.a_notch = signal.iirnotch(self.f0, self.Q, self.fs)
        self.b_band, self.a_band = signal.butter(self.order, [self.low, self.high], btype='band')
        self.zi_notch=[]
        self.zi_band=[]
        for i in range(70):
            self.zi_notch.append(signal.lfilter_zi(self.b_notch, self.a_notch))
            self.zi_band.append(signal.lfilter_zi(self.b_band, self.a_band))
        
        
        
        self.vhdr_fname='D:\\Downloads\\EEG_Data_03.vhdr'
        self.info_data=mne.io.read_raw_brainvision(self.vhdr_fname, misc='auto', scale=1000000.0)
        self.raw_data=self.info_data.get_data(start=0,stop=min(5000,self.info_data.n_times))
        self.filtered_data=self.Filter_data(self.raw_data)
        self.num_of_min_window=500
        self.num_idx=min(self.num_of_min_window,self.info_data.n_times)
        self.ref_elec=1
        
        
        
        #%%  Channel names and positions
        self.topomap_channels()

        #print(self.pos)
        
        
        
        
        
        
        #%% Adaptive filtering
        
#        self.adapt_filts=[]
#        for i in range(70):
#            self.adapt_filts.append(pa.filters.FilterNLMS(1, mu=0.00001 ,w='zeros'))
        self.weights=np.zeros(70)
        
            
        
        self.InitWindow()
        
        
        #%% Updating of windows
        
        self.update_graphs()
        self.ch_1.currentIndexChanged.connect(self.Choose_electrodes)
        self.ch_2.currentIndexChanged.connect(self.Choose_electrodes)
        self.freqs_1.currentIndexChanged.connect(self.Choose_frequencies)
        self.freqs_2.currentIndexChanged.connect(self.Choose_frequencies)
        
        
        self._timer_painter = QtCore.QTimer(self)
        self._timer_painter.start(500)
        self._timer_painter.timeout.connect(self.update_graphs) 
        
        
        for i in range(8):
            for j in range(8):
                self.Check[i][j].stateChanged.connect(self.idx_topomap_channels)
                
        self.ch_1_x.valueChanged[int].connect(self.Change_axes)
        self.ch_1_y.valueChanged[int].connect(self.Change_axes)
        self.ch_2_x.valueChanged[int].connect(self.Change_axes)
        self.ch_2_y.valueChanged[int].connect(self.Change_axes)
        
        
    def update_graphs(self):
        
        
        #%% Data reading
        #print(time.time())
        
        self.info_data=mne.io.read_raw_brainvision(self.vhdr_fname, misc='auto', scale=1000000.0)
        add_data=self.info_data.get_data(start=int(self.num_idx),stop=int(self.num_idx)+min(self.num_of_min_window,self.info_data.n_times-self.num_idx))
        self.num_idx+=min(self.num_of_min_window,self.info_data.n_times-self.num_idx)

        
        add_filtered_data=self.Filter_data(add_data)
        self.raw_data=np.concatenate((self.raw_data,add_data),axis=1)
        #adapted_data=self.Adaptive_cancelation(add_filtered_data,self.ref_elec,0.00000001)
        adapted_data=add_filtered_data
        self.filtered_data=np.concatenate((self.filtered_data,adapted_data),axis=1)
        if self.raw_data.size>10000*70:
            self.raw_data= self.raw_data[:,-5000:]
            self.filtered_data= self.filtered_data[:,-5000:]
        #print(time.time())
        
        
            
        freqs, welch_data=self.Welch_Freq(self.filtered_data)
        
        #print(time.time())
        
        
        self.data_1.clear()
        self.data_2.clear()
        self.freq_1.clear()
        self.freq_2.clear()
        
        
        self.data_1.plot(self.filtered_data[self.elec_1,-2000:], pen=pg.mkPen(color=(200, 0, 0)))
        self.data_2.plot(self.filtered_data[self.elec_2,-2000:], pen=pg.mkPen(color=(200, 0, 0)))
        self.freq_1.plot(welch_data[self.elec_1,:],pen=pg.mkPen(color=(200, 200, 0)))
        self.freq_2.plot(welch_data[self.elec_2,:],pen=pg.mkPen(color=(200, 200, 0)))
        
        
        self.freq_1.setXRange(10, 30, padding=0)
        self.freq_2.setXRange(10, 30, padding=0)
        self.freq_1.setYRange(0, self.freq_1_y,  padding=0)
        self.freq_2.setYRange(0, self.freq_2_y,  padding=0)
        
        
        self.data=[]
        self.ax[0].clear()
        self.ax[1].clear()
        for i in range(64):
            self.data.append(welch_data[i,self.welch_1])
            
        self.data=np.array(self.data)
        mne.viz.plot_topomap(self.data[self.idx_topo],self.pos[self.idx_topo],axes=self.ax[0],names=self.ch_names[self.idx_topo],show_names=True,contours=0)
        self.data=[]
        for i in range(64):
            self.data.append(welch_data[i,self.welch_2])
        self.data=np.array(self.data)
        mne.viz.plot_topomap(self.data[self.idx_topo],self.pos[self.idx_topo],axes=self.ax[1],names=self.ch_names[self.idx_topo],show_names=True,contours=0)
        
        self.figure_top.canvas.draw()
        #print(time.time())
        
        

        
                    
                    
                
        
        
        
        
        
        
        
        
        
        
        #%%
        
        
        
        
        
        # Initial GUI
    def InitWindow(self):
       # self.setWindowIcon(QtGui.QIcon("home.png"))

        #self.center()
        #montage=mne.channels.make_standard_montage('standard_1005')
        #print(montage)
        
        #self.info=mne.create_info(self.ch_names,500., montage=montage)
        #self.info.set_montage()
        self.data=[]
        
        for i in range(64):
            self.data.append(0.1)

        
        self.grid=QtGui.QGridLayout(self)
        widget = QtGui.QWidget(self)
        widget.setLayout(self.grid)
        #self.setLayout(self.grid)
        tabwidget=QTabWidget(self)
        tabwidget.addTab(widget, 'DATA')
        # BUTTONS DEFINING
        #grid.addWidget(self.createExampleGroup(), 1, 2, 1,1)
        
#        self.button_1=self.Create_Button(0,0,'start')
#        self.button_2=self.Create_Button(0,0,'end')
#        self.button_1.clicked.connect(self.Click_button)
#        self.grid.addWidget(self.button_1,2,0)
#        self.grid.addWidget(self.button_2,2,1)
        
        #plot = pg.PlotWidget()
        #plot1 = pg.PlotWidget()
        self.win = pg.GraphicsWindow()
        self.data_1=self.win.addPlot(0,0)
        self.data_2=self.win.addPlot(0,1)
        self.freq_1=self.win.addPlot(1,0)
        self.freq_2=self.win.addPlot(1,1)
        self.grid.addWidget(self.win,1,0)
        

        #grid.addWidget(plot1,1,1)
        #self.grid.setColumnStretch(1,1)
        self.groupBox = QGroupBox("Choose electrodes")
        self.Choose_channel = QVBoxLayout()
        self.Raw_1 = QHBoxLayout()
        self.ch_1_x = QSlider()
        self.ch_1_y = QSlider()
        self.ch_1 = QComboBox()
        self.Raw_1.addWidget(self.ch_1)
        self.Raw_1.addWidget(self.ch_1_x)
        self.Raw_1.addWidget(self.ch_1_y)
        self.Raw_2 = QHBoxLayout()
        self.ch_2_x = QSlider()
        self.ch_2_y = QSlider()
        self.ch_2 = QComboBox()
        self.Raw_2.addWidget(self.ch_2)
        self.Raw_2.addWidget(self.ch_2_x)
        self.Raw_2.addWidget(self.ch_2_y)

        #self.ch_1.setIconSize(QtCore.QSize(20, 10))
        self.ch_1.setFixedWidth(80)
        self.ch_2.setFixedWidth(80)
        self.ch_1_x.setFixedWidth(80)
        self.ch_2_x.setFixedWidth(80)
        self.ch_1_y.setFixedWidth(80)
        self.ch_2_y.setFixedWidth(80)
        self.ch_1.addItems(self.ch_names)
        self.ch_2.addItems(self.ch_names)
        self.Choose_channel.addLayout(self.Raw_1)
        self.Choose_channel.addLayout(self.Raw_2)
        self.groupBox.setLayout(self.Choose_channel)
        self.grid.addWidget(self.groupBox,2,0)
        
        
        
        
        self.groupBox_freqs = QGroupBox("Choose frequencies")
        self.Choose_freqs = QVBoxLayout()
        self.freqs_1 = QComboBox()
        self.freqs_2 = QComboBox()
        #self.ch_1.setIconSize(QtCore.QSize(20, 10))
        self.freqs_1.setFixedWidth(80)
        self.freqs_2.setFixedWidth(80)
        self.freqs_1.addItems(map(str,np.array(list(range(35)))))
        self.freqs_2.addItems(map(str,np.array(list(range(35)))))
        self.Choose_freqs.addWidget(self.freqs_1)
        self.Choose_freqs.addWidget(self.freqs_2)
        self.groupBox_freqs.setLayout(self.Choose_freqs)
        self.grid.addWidget(self.groupBox_freqs,2,1)
        

        
        
        

        
        
        #%%
        #ax= self.figure.add_subplot()

        #self.figure=mne.viz.plot_raw(a)
        #self.figure=a.plot_projs_topomap()
        #self.canvas = FigureCanvas(self.figure)
        
        #self.toolbar= NavigationToolbar(self.canvas, self)
        #grid.addWidget(self.canvas, 2,2,10,10)
        #grid.addWidget(self.toolbar, 0,0,1,1)
        #self.canvas.draw()
        
        self.figure_top, self.ax = plt.subplots(2, 1)
        mne.viz.plot_topomap(self.data,self.pos,axes=self.ax[0],names=self.ch_names,show_names=True,contours=0)
        mne.viz.plot_topomap(self.data,self.pos,axes=self.ax[1],names=self.ch_names,show_names=True,contours=0)
        
        self.canvas_top = FigureCanvas(self.figure_top)
        self.grid.addWidget(self.canvas_top, 1,1)
        
        
        
        self.grid_options=QtGui.QGridLayout(self)
        widget_options = QtGui.QWidget(self)
        widget_options.setLayout(self.grid_options)
        #self.setLayout(self.grid)
        tabwidget.addTab(widget_options, 'Options')
        self.grid_options.addWidget(self.Create_checkbox())
        
        




        self.show()
        
        
    def Filter_data(self, data):
        #  FILTERING DATA           
        # =============================================================================
        data, self.zi_notch = signal.lfilter(self.b_notch, self.a_notch, data, axis=1, zi=self.zi_notch)
        data, self.zi_band = signal.lfilter(self.b_band, self.a_band, data, axis=1, zi=self.zi_band)
        return data
    
    
    def Welch_Freq(self, data):
        f, w_data=signal.welch(data[:, -5000:],500,nperseg=500, axis=1)
        return f, w_data
    
    def Choose_electrodes(self):
        self.elec_1=self.ch_1.currentIndex()
        self.elec_2=self.ch_2.currentIndex()
        
    def Choose_frequencies(self):
        self.welch_1=self.freqs_1.currentIndex()
        self.welch_2=self.freqs_2.currentIndex()
        
      # CREATE BUTTON  
    def Create_Button(self,right,bottom,text):
        button=QPushButton(text, self)
        #button.move(right, bottom)
        #button.setGeometry(QRect(100,20,100,100))
        #button.setIcon(QtGui.QIcon('home.png'))
        #button.setIconSize(QtCore.Qsize(40,40))
        #button.setToolTip("This")
        return button
    
    def Adaptive_cancelation(self,data,ref_elec,mu):
        adapted=data
        for i in range (64):
            for j in range (500):
                adapted[i,j]=adapted[i,j]-self.weights[i]*data[ref_elec,j]
                delta_w=mu*adapted[i,j]*data[i,j]
                if abs(1-mu*data[i,j]*data[i,j])<1:
                    self.weights[i]+=delta_w
        #print(self.weights)

#        for i in range(70):
#            
#            idx=0
#            for j in range(500):
#                #print(self.adapt_filts[i].predict(j))
#                adapted[i,idx]=(data[i,idx]-self.adapt_filts[i].predict(j)[0])
#                self.adapt_filts[i].adapt(data[ref_elec,idx],data[i,idx])
#                idx+=1
#        print(len(adapted))
#        print(len(self.filtered_data))
        return adapted
    
    
    
    def Click_button(self):
        print("Hello")
        self.data=np.random.random(64)
        mne.viz.plot_topomap(self.data,self.pos,axes=self.ax[0],names=self.ch_names,show_names=True)
        self.figure_top.canvas.draw()
        
    def Create_checkbox(self):
        groupBox = QGroupBox("Electrodes for topomap")
        vbox = QtGui.QGridLayout()
        self.Check=[]
        for i in range(8):
            self.Check.append([])
            for j in range(8):
                self.Check[i].append(QtGui.QCheckBox(str(self.ch_names[8*i+j])))
                vbox.addWidget(self.Check[i][j],i,j)
                self.Check[i][j].setChecked(True)
        groupBox.setLayout(vbox)
        return groupBox
        
        
        
    def createExampleGroup(self):
            groupBox = QGroupBox("Best Food")

            radio1 = QRadioButton("&Radio pizza")
            radio2 = QRadioButton("R&adio taco")
            radio3 = QRadioButton("Ra&dio burrito")

            radio1.setChecked(True)

            vbox = QVBoxLayout()
            vbox.addWidget(radio1)
            vbox.addWidget(radio2)
            vbox.addWidget(radio3)
            vbox.addStretch(1)
            groupBox.setLayout(vbox)

            return groupBox
        
        
    def topomap_channels(self):
        print('hi')
        
        self.ch_names=['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','FCz','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','F2','AF4','AF8']
        self.ch_names=np.array(self.ch_names)
        EEG1005_layout = mne.channels.read_layout('EEG1005')
        pos_1005=EEG1005_layout.pos
        names_1005=EEG1005_layout.names
        self.pos=[]
        for i in range (len(self.ch_names)):
            for j in range (len(names_1005)):
                if self.ch_names[i]==names_1005[j]:
                    self.pos.append([pos_1005[j][0],pos_1005[j][1]])
        self.pos=np.array(self.pos)
        
    def idx_topomap_channels(self):
        print('hi')
        self.idx_topo=[]
        for i in range(8):
            for j in range(8):
                if self.Check[i][j].isChecked()==True:
                    self.idx_topo.append(int(i*8+j))
        self.idx_topo=np.array(self.idx_topo)
        print(self.idx_topo)
    
    
    
    def Change_axes(self):
        self.freq_1_y=self.ch_1_y.value()/10
        self.freq_2_y=self.ch_2_y.value()/10
        
        
        

        
if __name__ == "__main__":
    App=QApplication(sys.argv)
    window=Window()
    sys.exit(App.exec())


