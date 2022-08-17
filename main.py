# Hsuan-An Chen 2021/12/05
# change CNN model in line 64
# change SVM model in line 75
import os
import sys
import subprocess
import numpy as np

import music_record
import Music_emotion_models as Model
from Ui_Record_ui import Ui_MEanalysis

from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfile, askopenfilename

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, QTime, QDate, Qt 

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, source):
        self.filepath = os.getcwd()
        self.audiopath = self.filepath + '\\temp_audio.wav'
        # print(self.filepath)
        super(MainWindow, self).__init__()
        self.ui = Ui_MEanalysis()
        self.ui.setupUi(self)
        # timer variables
        self.flag = False
        self.count = 0

        self.source = source
        self.record_control = music_record.Control(self.source)

        self.SVM_model = Model.SVM()
        self.CNN_model = Model.CNN(ModelPath = 'music_cnn.model')
        
        self.ui.audio_name.setReadOnly(bool(self.ui.FileName_checkbox.checkState())!=True)
        self.ui.FilePath.setText(self.filepath)
        self.ui.FileName_checkbox.stateChanged.connect(self.AudioName)
        self.ui.browse.clicked.connect(self.Browse)
        
        self.ui.Start.clicked.connect(self.Start)
        self.ui.Stop.clicked.connect(self.Stop)

        self.ui.SVM_analyze.clicked.connect(self.SVM_predict)
        self.ui.CNN_analyze.clicked.connect(self.CNN_predict)
        
        timer = QTimer(self)  
        timer.timeout.connect(self.showTime) 
        timer.start(1000) 

        counter = QTimer(self)
        counter.timeout.connect(self.counter)
        counter.start(10)

    def CNN_predict(self):
        self.predicet_filepath = self.filepath + "\\" + self.ui.audio_name.text()+'.wav'
        # self.ui.Info.setText("")
        self.ui.Info.setText("[CNN predict！]  File > " + self.predicet_filepath)
        if os.path.isfile(self.predicet_filepath)==False:
            self.ui.Info.setText("You don't have the file！")
            return
        cnn_result = self.CNN_model.Predict('music_cnn.model', self.predicet_filepath) #change CNN model's name or path here.
        di = {1:'Joy' ,2: 'Tension' , 3:'Peacefulness' ,4:'Sadness'}
        self.ui.CNN_result.setText(di[cnn_result])

    def SVM_predict(self):
        # self.ui.Info.setText("")
        self.predicet_filepath = self.filepath + "\\" + self.ui.audio_name.text()+'.wav'
        self.ui.Info.setText("[SVM predict！]  File > " + self.predicet_filepath)
        if os.path.isfile(self.predicet_filepath)==False:
            self.ui.Info.setText("You don't have the file！")
            return
        svm_result  = self.SVM_model.Predict('music00', self.predicet_filepath) #change SVM model's name or path here.     
        di = {1:'Joy' ,2: 'Tension' , 3:'Peacefulness' ,4:'Sadness'}
        self.ui.SVM_result.setText(di[svm_result[0]])

    def showTime(self): 
        current_date = QDate.currentDate()
        current_time = QTime.currentTime()
        label_time = "Date： " + current_date.toString('yyyy/MM/dd') + " " + current_time.toString('hh:mm:ss')
        self.ui.clock.setText(label_time) 

    def counter(self):
        # if self.count >= 1500:
        #     self.ui.Info.setText("Automatically Stop -> It's over 15 sec！")
        #     self.Stop()
        if self.flag:   
            self.count+= 1
        if self.count==0:
            text = " 00:00 s"
        elif self.count!=0:
            text = "  " + str(int(self.count / 100)) + ":" + str(int(self.count % 100)) + " s"
        self.ui.RecordTime_bar.setValue(self.count / 100)
        self.ui.counter.setText(text)
  
    def AudioName(self):
        print(bool(self.ui.FileName_checkbox.checkState()))
        if bool(self.ui.FileName_checkbox.checkState()):
            self.ui.audio_name.setReadOnly(bool(self.ui.FileName_checkbox.checkState())!=True)
        else:
            self.ui.audio_name.setText('temp_audio')
            self.ui.audio_name.setReadOnly(bool(self.ui.FileName_checkbox.checkState())!=True)
        
    def Browse(self):
        Tk().withdraw()
        temp = self.filepath
        self.filepath = askdirectory(initialdir=self.filepath)
        if bool(self.filepath)==False:
            self.filepath = temp
        self.audiopath = self.filepath + "\\" + self.ui.audio_name.text()+'.wav'
        self.ui.FilePath.setText(self.filepath)
  
    def Start(self):
        self.ui.Info.setText("")
        self.ui.SVM_result.setText("")
        self.ui.CNN_result.setText("")
        self.ui.Info.setText("Start Recording！")
        self.audiopath = self.filepath + "\\" + self.ui.audio_name.text()+'.wav'
        if self.audiopath == self.filepath + "\\" + 'temp_audio.wav':
            os.remove(self.audiopath)
        elif self.audiopath != self.filepath + "\\" + 'temp_audio.wav':
            if os.path.isfile(self.audiopath):
                print("\nerror: File exist！\n")
                self.ui.Info.setText("File exist！")
                return
        if bool(self.ui.FileName_checkbox.checkState()):
                self.audio_file = self.ui.audio_name.text()+'.wav'
                self.audiopath = self.filepath + "\\" + self.audio_file
        else:
            self.audiopath = self.filepath + '\\temp_audio.wav'
        self.count = 0 
        self.record_control.StartRecord(self.audiopath)
        self.flag = True
  
    def Stop(self):
        self.ui.Info.setText("")
        self.ui.Info.setText("End Recording！")
        if self.flag == False:
            self.ui.Info.setText("You need to start first！")
            return
        if self.count <= 400:
            self.ui.Info.setText("You should record more than 4 sec！")
            return
        self.count = 0
        self.flag = False
        self.record_control.StopRecord(self.audiopath)
        self.ui.Info.setText("Save file to > " + self.audiopath)

## Main.py ##    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow('source/')
    window.setGeometry(400, 200, 600, 700)
    window.show()
    sys.exit(app.exec_())