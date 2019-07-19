from pydub import AudioSegment
from pathlib import Path
import math
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

def one_hot_encode(labels, num_classes):
    """ Outputs one-hot encoding matrix (matrix of ones/zeros) for
        given list.

        Inputs:
           labels: list/array, contains labels/tags, shape=(1,num_examples)
           num_classes: int, number of different labels/tags
        Outputs:
           y_encoded: matrix containing one-hot encodings for labels
    """

    m = len(labels) #number of examples
    y_encoded = np.zeros((m, num_classes))

    for i in range(m):
        y_encoded[i, int(labels[i][0]) - 1] = 1


    return y_encoded

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

def preproc_data(df, col_name, train_pt = 0.80):
    """Renames fname entries & splits the data into train/dev according to the percentage
    split train_pt
    
    Inputs:
        df - pandas dataframe, contains the audio_data with fname, label cols
        col_name - str, name of the column for the filenames
        train_pt - float (between 0 and 1), percent of data set for training  
    """
    num_train = train_pt * len(df[col_name]) 
    for i in range(len(df[col_name])):
        if i <= num_train:
            df.fname[i] = "train/" + df[col_name][i]
        else:
            df.fname[i] = "valid/" + df[col_name][i]

def snip_data(folder, num_classes, remlist, song_lengths, offset):
    
    folder_list = listdir(folder) 
    for i in range(1, num_classes):
        sel = []
        sel = [f for f in folder_list if f.startswith(str(i))]
        print(offset)
        for e in sel:
            if e.endswith('clip' + str(song_lengths[i-1] - (len(sel) + offset)) + '.wav') or e.endswith('clip' + str(len(sel)-1) + '.wav'):
                remlist.append(e)
                
    
    for j in range(len(remlist)):
         os.remove(folder+str(remlist[j]))
           
    assert(folder_list[i] != remlist[j] 
           for i in range(len(folder_list)) for j in range(len(remlist)))

def generate_melspecs(input_dir, ms_output, n_fft=1024, n_hop=256, n_mels=40, fmin=20,
                      fmax=8000, my_dpi=96):
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
    
    temp_list = listdir(input_dir)
    input_list = col_names + [[temp_list[i], temp_list[i][0]] for i in range(len(temp_list))]
    with open(csv_name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(input_list)
        

def move_wav_to_tvfolders(folder, df, ext):
    for i in range(len(df.fname)):
        for file in listdir(folder):
            if file.endswith(ext) and file in df.fname[i]:
                os.rename(folder/file, folder/df.fname[i])

def clear_out(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)

def initialise_data(img_folder, image_shape):
    num_specs_train = len(os.listdir(img_folder))
    X_train = np.zeros((num_specs_train, 
                    image_shape[0],
                    image_shape[1],
                    image_shape[2]))
    y_train = np.zeros((num_specs_train, 1))
    
    return X_train, y_train

def gen_input_and_label_array(img_folder, df, image_shape):
    
    X, y = initialise_data(img_folder, image_shape)
    
    for j in range(len(df)):
        name = df.fname[j]
        label = df.label[j]
        img = imageio.imread(img_folder + name)
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
        import random
        for j in range(math.floor(test_frac * len(os.listdir(ms_dir)))):
            x = random.sample(os.listdir(ms_dir), 1)
            shutil.move(ms_dir + x[0], test_ms_dir)
        
    home = Path('.')

    gen_csv_file(ms_dir, csv_name, col_names)
    IMG_CSV = home / csv_name

    ms_df = pd.read_csv(IMG_CSV)
    ms_df = ms_df.sample(frac=1).reset_index(drop=True)

    X, y = gen_input_and_label_array(ms_dir, ms_df, img_shape)
    X = X / 255
    
    return X, y, ms_df

