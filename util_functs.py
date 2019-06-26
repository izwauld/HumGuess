from pydub import AudioSegment
from pydub.playback import play
import math
import numpy as np
import re
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

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


def one_hot_encode(labels, num_classes):
    """ Outputs one-hot encoding matrix (matrix of ones/zeros) for
        given list.

        Inputs:
           labels: list, contains labels/tags, shape=(1,num_examples)
           num_classes: int, number of different labels/tags
        Outputs:
           y_encoded: matrix containing one-hot encodings for labels
    """

    m = len(labels) #number of examples
    y_encoded = np.zeros((num_classes, m))

    for i in range(m):
        y_encoded[labels[i], i] = 1


    return y_encoded


def extract_label(song, class_num):
    """ Finds the label of the song using regular expression.

        Inputs:
           song: str, audio clip
           class_num: int, song label (0-5)
    """
    lab = re.findall('\A' + str(class_num), song)
    label = lab[0]
    return label


def create_audio_clips(input_path, out_path, label):
    """ Creates a series of audio clips using the audio_splice method,
         automatically generating clips from each peron's song.

         Inputs:
           input_path: str, path directory
           output_path: str, path where clips are stored
           label: int, category label (0-5)
    """
    file_list = listdir(input_path)

    for i in range(len(file_list)):
        audio_splice(input_path + file_list[i], out_path + str(label) + "_"
                     + str(i) + "_", 10, 8)

