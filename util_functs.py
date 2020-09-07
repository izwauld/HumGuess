from pydub import AudioSegment
from pathlib import Path
import math, random
import numpy as np
import pandas as pd
import os
import csv
import shutil
from os import listdir
from os.path import isfile, join, splitext
import librosa, librosa.display
from scipy.io import wavfile
import imageio
from scipy.fftpack import fft
from scipy.signal import get_window
import matplotlib.pyplot as plt

LABEL_DICT = { 0 : 'None', 1 : 'Michael Jackson - Thriller', 2 : 'Survivor - Eye of the Tiger', 
              3 : 'One Direction - What Makes You Beautiful', 
              4 : 'Queen - We Are The Champions', 5 : 'Frozen - Let It Go'}

def audio_splice(song_path, output_path, clip_length, stride):
    """Ouputs series of audio clips (defined by CLIP_LENGTH) from a
       longer input audio file (song_path)

       Inputs:
          song_path: path to audio file
          output_path: path for generated clips
          clip_length: length of audio file, seconds
          stride: step size when creating clips, int
    """

    song = AudioSegment.from_wav(song_path) #extract audio

    STRIDE = stride * 1000 #pydub works in ms
    CLIP_LENGTH = clip_length * 1000 #pydub works in ms
    num_clips = math.floor((len(song) - CLIP_LENGTH) / STRIDE) + 1

    for t in range(num_clips):
        clip = song[STRIDE*t:CLIP_LENGTH + STRIDE*t]
        clip.export(output_path + "clip" + str(t) + ".wav", format="wav")

    return num_clips

def create_audio_clips(input_path, out_path, clip_length=10, stride=8):
    """ Creates a series of audio clips using the audio_splice method,
        (10s clips with stride=8), automatically generating clips from each peron's song.

         Inputs:
           input_path: str, path to full song recordings
           output_path: str, path where clips are stored
    """
    clips = listdir(input_path)
    
    #file_list = [f for f in clips if f.endswith('.wav')]
    for elem in clips:
        label = splitext(elem)[0][-1]
        initial = splitext(elem)[0][0]
        audio_splice(input_path + elem, out_path + label + "_"
                     + initial + "_", clip_length, stride)

def generate_melspecs(input_dir, ms_output, n_fft=1024, n_hop=256, n_mels=40, fmin=20,
                      fmax=8000, my_dpi=96):
    """Creates the melspectogram images associated with the input audio clips.
    
    Inputs:
          input_dir: path, location of the input audio clips
          ms_output: path, desired location of the melspectograms
          n_fft: int, number of bins for spectogram in which frequency information is collected
          n_hop: int, hop length determining how many samples we slide along by
          n_mels: int, number of frequency bins for the mel-spectogram (different from n_fft)
          fmin: int, minimum frequency in the melspectogram
          fmax: int, maximum frequency in the melspectogram
          my_dpi: int, dpi for the images
          
    """
    mel_spec_array = []
    label_array = []
    fname_array = []
    inputs = listdir(input_dir)
    
    file_list = [f for f in inputs if f.endswith('.wav')]
    for elem in file_list:
        fname = splitext(elem)[0]
        label = elem[0]
        
        clip, sample_rate = librosa.load(input_dir+elem, sr=None)
        mel_spec = librosa.feature.melspectrogram(clip, n_fft=n_fft, hop_length=n_hop,
                                          n_mels=n_mels, sr=sample_rate, power=1.0, 
                                          fmin=fmin, fmax=fmax)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        mel_spec_array.append(mel_spec_db)
        label_array.append(label)
        fname_array.append(fname)
        
    for i in range(len(mel_spec_array)):
        plt.figure(figsize=(50/my_dpi, 34/my_dpi), dpi=my_dpi) #(34,50)
        librosa.display.specshow(mel_spec_array[i],
                                 sr=sample_rate, hop_length=n_hop,
                                 fmin=fmin, fmax=fmax)
        plt.savefig(ms_output + fname_array[i])
        plt.close()
        
    #return mel_spec_array, label_array, fname_array

def gen_csv_file(input_dir, csv_name, col_names = [['fname', 'label']]):
    """Generates a csv file from contents of a directory (input_dir
    with a given name (csv_name)

    Inputs:
       input_dir - str, directory you want to make csv file from
       csv_name - str, name of the csv file (***NEEDS .CSV EXTENSION***)
       col_names - list, tuple containing column names for csv
    """
    
    temp_list = listdir(input_dir) #lists the filenames of the contents of input_dir
    input_list = col_names + [[temp_list[i], temp_list[i][0]] for i in range(len(temp_list))] #Here, temp_list[i] is file name / temp_list[i][0] is the label
    with open(csv_name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(input_list)
        
def initialise_data(img_folder, image_shape):
    """Defines the training exmaples/labels based on the shape of the
    input images and the image folder.
    
    Inputs:
          img_folder: path, where the melspectograms are located (defines # of examples)
          image_shape: tuple, the dimensions of the input images
    Outputs:
          X_train: array, null array with dimensions (num_examples, image_shape)
          y_train: array, null target array with dimensions (num_examples, 1)
    """
    num_specs_train = len(os.listdir(img_folder))
    X_train = np.zeros((num_specs_train, 
                    image_shape[0],
                    image_shape[1],
                    image_shape[2]))
    y_train = np.zeros((num_specs_train, 1))
    
    return X_train, y_train

def gen_input_and_label_array(img_folder, df, image_shape):
    """Populates the input and label arrays initialised prior with
    image tensors and image labels, respectively.
    
    Inputs:
          img_folder: path, location of the mel-spectograms
          df: Pandas dataframe, dataframe containing clip name/label information
          image_shape: tuple, shape of the melspectogram images
    Outputs:
          X: array, contains traing examples, shape = (# of examples, image_shape)
          y: array, conatains labels for training inputs, shape = (# of examples, 1)
    """
    X, y = initialise_data(img_folder, image_shape)
    
    for j in range(len(df)):
        name = df.fname[j] #name of clip comes from 'fname' column in the dataframe
        label = df.label[j] #label of clip comes from 'label' column in the dataframe
        img = imageio.imread(img_folder + name) #numpy array containing image pixel values
        #Assign img array to the jth training example, assign label to the jth label
        X[j] = img
        y[j] = label

    return X, y    

def generate_data(img_shape, num_classes = 5, cl = 5, s = 1.5, n_fft = 1024, n_hop = 256, n_mels = 40,
                  fmin = 20, fmax = 8000, createClips = False, genMelSpecs = False, storeTest = False, test_frac = 0.10,
                  master_audio_dir = 'input_songs/', ac_dir = 'input_clips/', ms_dir = 'input_ms/',
                  test_ms_dir = 'test_ms/', csv_name = 'msDataFile.csv', col_names =[['fname', 'label']]):
    """Returns the training data and labels needed to pass to the
    model

    Inputs:
       img_shape - tuple, shape of the melspectrograms
       cl - int/float, clip length
       s - int/float, stride taken when generating audio clips
       createClips - bool, set True if you want to create new audio
          clips with custom length and stride. False if you already have clips
          and just need the melspectrograms
       test_frac - float, fraction of the training data you want to store
          away for testing the model after it has finished on train/valid data
       csv_name - string, name of the csv file for storing names
           and labels for the mel-spectrogram dataframe
       master_audio_dir - string, location of the raw, uncut song .wavs
       ac_dir - string, name of the directory where the audio
           clips generated from create_audio_clips are stored
       ms_dir - string, name of the directory where the mel-specs
           generated from generate_melspecs are stored
       col_names - list, name of the columns that will appear
           in the mel-spectrogram dataframe 

    Outputs:
       X - array, dataset containing the melspectrogram info,
          shape = (num_of_spectrograms, img_shape)
       y - array, labels for the melspectrogram,
          shape = (num_of_spectrograms, 1)
       ms_df - pandas dataframe, contains the filename + label data for
       all mel-spectrograms in the ms_dir directory

    """
    if createClips:
        create_audio_clips(master_audio_dir, ac_dir, cl, s)
        
    if genMelSpecs:
        generate_melspecs(ac_dir, ms_dir)

    if storeTest:
      #Move the amount of test specified away to the test directory, test_ms_dir
        for j in range(math.floor(test_frac * len(os.listdir(ms_dir)))):
            #Randomly choose which examples are stored as test
            x = random.sample(os.listdir(ms_dir), 1)
            shutil.move(ms_dir + x[0], test_ms_dir)
        
    home = Path('.')
    
    #Generate csv file for mel-specs & define path to that csv
    gen_csv_file(ms_dir, csv_name, col_names)
    ms_csv_path = home / csv_name
    
    #Read the mel-spec csv into a Pandas dataframe & reformat the indices
    ms_df = pd.read_csv(ms_csv_path)
    ms_df = ms_df.sample(frac=1).reset_index(drop=True)

    X, y = gen_input_and_label_array(ms_dir, ms_df, img_shape)
    #Normalise the image pixels to the range (0,1)
    X = X / 255
    
    return X, y, ms_df
