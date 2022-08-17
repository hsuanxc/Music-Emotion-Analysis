#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('git clone https://github.com/IanChen5273/Music-emotion.git')


# In[1]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "請填入自己組別的GPU 代號"
import tensorflow as tf
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus[0])
if gpus:
  	# Restrict TensorFlow to only use the first GPU
	try:
		tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
		tf.config.experimental.set_memory_growth(gpus[0], True)
	except RuntimeError as e:
		# Visible devices must be set at program startup
		print(e) 


# In[2]:


import joblib
import librosa
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, add, Flatten, Dropout, Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import xlrd

# In[3]:


#DataPath = '/workspace/Project1/Music-emotion/music-emotion/song/'
#os.chdir(DataPath)
#xlsx_module = xlrd.open_workbook('MusicData.xlsx')
#xlsx_sheet = xlsx_module.sheets()[0]
#data_info = []
#for i in range(1, xlsx_sheet.nrows):
#    data_info.append(xlsx_sheet.row_values(i))


# In[4]:


class SVM:
    def __init__(self):
        pass

    def Train(self, DataPath):
        """Train data with SVM model

        Parameters
        ----------
        DataPath : string
            The path where you store the data. Note that the path must contain every split music data and
            MusicData.xlsx
        """
        Features, Labels = self.LoadFeature(DataPath)

        #MeanValue = np.array(np.mean(Features, axis=0))
        #DiffValue = np.array(np.max(Features, axis=0) - np.min(Features, axis=0))
        #Features = (Features - MeanValue) / DiffValue

        X_train, X_test, y_train, y_test = train_test_split(Features, Labels, test_size=0.1, random_state=42)

        svm_model = svm.SVC(kernel='rbf', C=1, gamma='scale')
        svm_model.fit(X_train, y_train.ravel())
        joblib.dump(svm_model, 'music00')

        print('train = ', svm_model.score(X_train, y_train))
        print('val = ', svm_model.score(X_test, y_test))

    def Predict(self, ModelPath, InputData):
        """Predict the input with the given model

        Parameters
        ----------
        ModelPath : string
            The path where you store the model.
        InputData : string
            The path of input data, predict one data at one time.
        """
        X_test = self.FeatureExtraction(InputData)
        loaded_model = joblib.load(ModelPath)
        result = loaded_model.predict(X_test)
        di = {1:'Joy' ,2: 'Tension' , 3:'Peacefulness' ,4:'Sadness'}
        print(di[result[0]])

        return result

    def FeatureExtraction(self, DataPath):
        """Extract the feature for the given file

        Parameters
        ----------
        DataPath : string
            The path where your data is, only allows the file of wav and wma format.
        """
        y, sr = librosa.load(DataPath, sr=22050, mono=True, duration=4)
        time = librosa.get_duration(y=y, sr=sr)
        if int(time) < 4:
            return None

        feature_mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=sr, n_mfcc=1)
        feature_mfcc = feature_mfcc.reshape((1, feature_mfcc.shape[1] * feature_mfcc.shape[0]))
        feature_rolloff = librosa.feature.spectral_rolloff(y=y, sr=22050, hop_length=sr, roll_percent=0.85)
        feature_spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=22050, hop_length=sr, n_bands=1)
        feature_spectral_contrast = feature_spectral_contrast.reshape(
            (1, feature_spectral_contrast.shape[1] * feature_spectral_contrast.shape[0]))
        feature_rms = librosa.feature.rms(y=y, hop_length=sr)
        feature_combine = np.hstack((feature_rms, feature_mfcc))
        feature_combine = np.hstack((feature_combine, feature_rolloff))
        feature_combine = np.hstack((feature_combine, feature_spectral_contrast))

        return feature_combine

    def LoadFeature(self, DataPath):
        """Load features and labels from the directory.

        Parameters
        ----------
        DataPath : string
            The path where you store the data. Note that the path must contain every split music data and
            MusicData.xlsx
        """
        os.chdir(DataPath)
        xlsx_module = xlrd.open_workbook('MusicData.xlsx')
        xlsx_sheet = xlsx_module.sheets()[0]
        data_info = []
        for i in range(1, xlsx_sheet.nrows):
            data_info.append(xlsx_sheet.row_values(i))

        features = np.array([])
        music_label = []
        for i in range(len(data_info)):
            print("\rExtracting feature ({}/{})".format(i, len(data_info)), flush=True, end='')
            print(" : {}/{}".format(data_info[i][1], data_info[i][2] + data_info[i][5]), flush=True, end='')

            os.chdir(data_info[i][1])

            feature = self.FeatureExtraction(data_info[i][2] + data_info[i][5])
            os.chdir('..')

            if feature is None:
                continue

            if features.shape[0] == 0:
                features = feature
            else:
                features = np.vstack((features, feature))

            arousal_s = data_info[i][3] - 50
            valence_s = data_info[i][4] - 50
            if arousal_s >= 0 and valence_s >= 0:  # Joy
                music_label.append(0)
            elif arousal_s >= 0 and valence_s < 0:  # Tension
                music_label.append(1)
            elif arousal_s < 0 and valence_s >= 0:  # Peacefulness
                music_label.append(2)
            elif arousal_s < 0 and valence_s < 0:  # Sadness
                music_label.append(3)

        os.chdir('..')
        print('\rExtract complete', flush=True)

        return features, np.array(music_label)


# In[5]:


#model = SVM()
# model.Train('/workspace/Project1/Music-emotion/music-emotion/song')
#result = model.Predict('/home/cils0003/Desktop/music_ui-20211125T053746Z-001/music_ui/music00','/home/cils0003/Desktop/music_ui-20211125T053746Z-001/music_ui/UI_records/temp_audio.wav')
#print(result)


# In[6]:


import warnings
warnings.filterwarnings("ignore")
class CNN:
    def __init__(self, ModelPath, epoch=10, batch_size=8, verbose=1, sr=22050):
        self.org_path = os.getcwd()
        self.set_epoch = epoch
        self.set_batch_size = batch_size
        self.set_verbose = verbose
        self.sr = sr
        self.model = load_model(ModelPath)
    def LoadData(self, DataPath):
        """Load data from the directory.

        Parameters
        ----------
        DataPath : string
            The path where you store the data. Note that the path must contain every split music data and
            MusicData.xlsx
        """
        os.chdir(DataPath)
        xlsx_module = xlrd.open_workbook('MusicData.xlsx')
        xlsx_sheet = xlsx_module.sheets()[0]
        data_info = []
        for i in range(1, xlsx_sheet.nrows):
            data_info.append(xlsx_sheet.row_values(i))
        music_data = []
        music_label = []
        data_len = []
        for i in range(len(data_info)):
            # print("\rLoading data ({}/{})".format(i, len(data_info)), flush=True, end='')
            # print(" : {}/{}".format(data_info[i][1], data_info[i][2] + data_info[i][5]), flush=True, end='')

            os.chdir(data_info[i][1])
            # raw_data, sr = librosa.load(data_info[i][2] + data_info[i][5], sr=22050, mono=True, offset=0.0, duration=4)
            raw_data, sr = librosa.load(data_info[i][2] + data_info[i][5], sr=self.sr, mono=True, offset=0.0, duration=4)
            music_data.append(raw_data)
            data_len.append(raw_data.size)
            arousal_s = data_info[i][3] - 50
            valence_s = data_info[i][4] - 50
            if arousal_s >= 0 and valence_s >= 0:  # Joy
                music_label.append(0)
            elif arousal_s >= 0 and valence_s < 0:  # Tension
                music_label.append(1)
            elif arousal_s < 0 and valence_s >= 0:  # Peacefulness
                music_label.append(2)
            elif arousal_s < 0 and valence_s < 0:  # Sadness
                music_label.append(3)
            os.chdir('..')
        music_label = np.array(music_label)
        data_len = np.array(data_len)

        last_index = np.where(data_len == sr * 4)[0]
        new_music_data = []
        new_music_label = []
        for i in range(last_index.size):
            new_music_data.append(music_data[last_index[i]])
            new_music_label.append(music_label[i])
        new_music_data = np.array(new_music_data)
        new_music_label = np.array(new_music_label)

        new_music_data = np.reshape(new_music_data, [new_music_label.size, 1, self.sr*4, 1])
        train_index, test_index = self.split_data(new_music_label, train_size=0.7)
        train_data = new_music_data[train_index, :, :, :]
        train_label = to_categorical(new_music_label[train_index])
        test_data = new_music_data[test_index, :, :, :]
        test_label = to_categorical(new_music_label[test_index])

        return train_data, train_label, test_data, test_label

    def split_data(self, label, train_size=0.7):
        import random
        uni_label = np.unique(label)
        uni_label_num = [[] for i in range(uni_label.size)]
        train_index = np.array([], dtype='uint32')
        for i in range(uni_label.size):
            uni_label_num = list(np.where(label == uni_label[i])[0])
            train_index = np.hstack(
                [train_index, np.array(random.sample(uni_label_num, k=int(len(uni_label_num) * train_size)))])
        train_index = np.sort(train_index)
        test_index = np.setdiff1d(np.arange(label.size), train_index)

        return train_index, test_index

    def CreateModel(self, InputShape):
        """Create CNN model

        Parameters
        ----------
        InputShape :
            The shape of training data
        """
        sig_input = Input(shape=InputShape)
        x1_1=Conv2D(128,(1,256),kernel_initializer='random_uniform',padding='same',data_format='channels_last')(sig_input)
        x1_1=(Activation('relu'))(x1_1)
        x1_1=(BatchNormalization())(x1_1)
        x1_1=MaxPooling2D((1,80),strides=(1,40),data_format='channels_last')(x1_1)
        x1_2=Conv2D(128,(1,512),kernel_initializer='random_uniform',padding='same',data_format='channels_last')(x1_1)
        x1_2=(Activation('relu'))(x1_2)
        x1_2=(BatchNormalization())(x1_2)
        x1_o=add([x1_1,x1_2])
        x2=Conv2D(32,(1,1024),kernel_initializer='random_uniform',padding='same',data_format='channels_last')(x1_o)
        x2=(Activation('relu'))(x2)
        x2=(BatchNormalization())(x2)
        x2=MaxPooling2D((1,80),strides=(1,40),data_format='channels_last')(x2)
        y=(Flatten())(x2)
        y=(Dropout(0.4))(y)
        y=(Dense(units=64,activation='relu'))(y)
        y=(Dropout(0.4))(y)
        output=(Dense(units=4,activation='softmax'))(y)
        model = Model(sig_input, output)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def Train(self, DataPath):
        """Train data with CNN model

        Parameters
        ----------
        DataPath : string
            The path where you store the data. Note that the path must contain every split music data and
            MusicData.xlsx
        """
        train_data, train_label, test_data, test_label = self.LoadData(DataPath)
        model = self.CreateModel(train_data.shape[1:])

        train_history = model.fit(x=train_data,
                                  y=train_label,
                                  validation_data=(test_data, test_label),
                                  epochs=self.set_epoch,
                                  batch_size=self.set_batch_size,
                                  verbose=self.set_verbose)
        os.chdir(self.org_path)
        model.save('music_cnn.model')

        # %% Model accuracy
        y1 = train_history.history['acc']
        y2 = train_history.history['val_acc']
        x = np.arange(len(y1)) + 1
        plt.plot(x, y1, color='blue', label='Train')
        plt.plot(x, y2, color='red', label='Validation')
        plt.title("accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='upper left', shadow=True)
        plt.savefig('accuracy', dpi=1500)
        plt.show()

        # %% Model loss value
        y1 = train_history.history['loss']


        y2 = train_history.history['val_loss']
        x = np.arange(len(y1)) + 1
        plt.plot(x, y1, color='blue', label='Train')
        plt.plot(x, y2, color='red', label='Validation')
        plt.title("loss")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.legend(loc='upper right', shadow=True)
        plt.savefig('loss', dpi=1500)
        plt.show()

    def Predict(self, ModelPath, InputData):
        """Predict the input with the given model

        Parameters
        ----------
        ModelPath : string
            The path where you store the model.
        InputData : string
            The path of input data, predict one data at one time.
        """
        raw_data, sr = librosa.load(InputData, sr=self.sr, mono=True, offset=0.0, duration=4)
        #model = load_model(ModelPath)
        music_data = np.reshape(raw_data, [1, 1, self.sr*4, 1])
        score = self.model.predict(music_data)
        pre_label = np.argmax(score)
        di = {1:'Joy' ,2: 'Tension' , 3:'Peacefulness' ,4:'Sadness'}
        print(di[pre_label])

        return pre_label


# In[7]:


#model = CNN()
# model.Train(DataPath)
#result = model.Predict('/home/cils0003/Desktop/music_ui-20211125T053746Z-001/music_ui/music_cnn.model','/home/cils0003/Desktop/music_ui-20211125T053746Z-001/music_ui/UI_records/temp_audio.wav')
#print(result)


# In[ ]:




