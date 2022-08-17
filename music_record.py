import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtCore import QTimer, QTime, Qt 
import pyaudio
import wave
import threading
import time
import subprocess
import os


class Control:
    def __init__(self, source):
        self.source = source
        # self.filename = "temp_audio.wav"
        self.audio_thread = None

    def StartRecord(self, audio_path):
        print('recording .......')
        self.audio_thread = AudioRecorder(audio_path)
        self.audio_thread.start()
        print("Press Pause to stop recording")


    def StopRecord(self, audio_path):
        print('saving files .....')
        self.audio_thread.stop()
        print(f"Audio saved to {audio_path}")

#Command: {arecord -D hw:2,0 --dump-hw-params}  for more audio detail              

class AudioRecorder():

    # Audio class based on pyAudio and Wave
    def __init__(self, audio_filename="temp_audio.wav"):

        self.open = True
        self.rate = 48000
        self.device_index = None
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = audio_filename
        self.file_manager()
        self.audio = pyaudio.PyAudio()  
        self.find_device()   
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      #input_device_index=int(self.device_index),
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):
        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break


    # Finishes the audio recording therefore the thread too    
    def stop(self):

        if self.open==True:
            self.open = False
            print(f"[AUDIO] Record last for {time.time() - self.start} sec")
            print(f"[AUDIO] Total audio size: {len(self.audio_frames)}")
            time.sleep(1)
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

        

    # Launches the audio recording function using a thread
    def start(self):
    
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()
        self.start = time.time()
        print("[AUDIO] Record Started")
        
    # Check if audio file exist
    def file_manager(self):

        local_path = os.getcwd()
        if os.path.exists(str(local_path) + self.audio_filename):
            os.remove(str(local_path) + self.audio_filename)
    
    # Find mic device in jetson nano        
    def find_device(self):
    
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                device_name = self.audio.get_device_info_by_host_api_device_index(0, i).get('name')
                print ("Input Device id ", i, " - ", device_name)
                finds = device_name.split(" ")
                for find in finds:
                    if find == "Audio":
                        self.device_index = i
                if self.device_index:
                    print(f"[AUDIO] Selected Device: {self.device_index}")  
                    break   
#if __name__ == "__main__":
    
#     filename = "temp_audio.wav"
#     audio_thread = AudioRecorder(filename)
#     audio_thread.start()
#     print("[Sys] Input s or S to stop recording")
#     cmd = input()
#     if cmd == "s" or "S":
#       audio_thread.stop()
#       print(f"[Sys] Audio saved to {filename}")
