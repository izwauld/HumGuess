# HumGuess - Multi-class classification ML project

In this repository, you will find helpful notebooks detailing the HumGuess project - where a machine learning approach is used to
predict what song someone is singing/humming. The approach used was inspired by work done by John Hartquist on an experimental [fast.ai audio classification module](https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89), and by a CNN approach to [classifiying MNIST digits](https://medium.com/x8-the-ai-community/audio-classification-using-cnn-coding-example-f9cbd272269e). For a more techincal insight into the methods used, please consult these resources. $$\LaTeX$$

## Table of Contents

1. [Overview](https://github.com/izwauld/HumGuess#overview)- Motivating the problem
2. [Setup](https://github.com/izwauld/HumGuess#setup) - Installing tools
3. [Data Generation](https://github.com/izwauld/HumGuess#data-generation) - Methodology behind data generation
4. [Running the CNN](https://github.com/izwauld/HumGuess#running-the-cnn) - Introducing CNN architecture, step-by-step
5. [Closing Thoughts](https://github.com/izwauld/HumGuess#closing-thoughts) - Review of approach, future directions to explore

## Overview

Have you ever been listening to music, and come across a song that you really like the sound of, but you don't know the name of it? You might pick up on some of the lyrics of the song, but it isn't enough to be able to know what the song name is. If the song is popular enough, you may be able to type in the lyrics into some search engine and it will fetch the song for you, but that isn't always a reliable way to get the name of the song. If you play the song to Shazam or Google, it might be able to recgonise it for you. But now it's too late - the song has passed. All you have to go off of is the beat that is stuck in your head.

So, let's harness that!

The purpose of this project is to build a machine learning system that will be able to predict what song you are singing or humming. For this project, I framed the problem as a supervised, multi-label classification problem. Basically, The ML model is trained to predict one of five songs, where you input an audio clip into the ML algorithm, and it spits out a probability that the clip refers to one of the five songs.

The model is trained on only one accent currently, just to demonstrate that this sort of approach can work. Ideally you would like to train on many voices to make the model more robust, but in fact, the guiding intuition is that the model doesn't really need to be trained on that many voices, just a more varied selection. This is because of the song choice: the songs chosen for this project are easily recognisable (like Michael Jackson - Thriller), so intuitively the frequency patterns of the audio clips should be similar regardless of the singer's accent. This means that you don't necessarily have to train on million's of voices! 

One major drawback to this approach is scalability: there are millions of songs out there, and more being released every day. This means that in order for this approach to be super robust, the model would have to be able to correctly identify millions of songs, but even if you got that to work, it would need to be retrained every time to account for new songs being released. So clearly, that isn't feasible!

With the high-level introduction out the way, we move onto the setup, where we introdcue the packages and dependencies that are needed.


## Setup

The following tools/dependencies were used in the project:
* [Python] - version 3.7.3
* [Tensorflow] - high-level machine learning framework, version 1.14.0
* [pydub] - audio package for Python, version 0.23.1
* [librosa] - audio package for Python, version 0.6.3

## Data Generation

The project pipeline looks like this:

As can be seen, the first step is to create the input data, namely the mel-spectrograms that will be fed into the CNN. All operations are sheltered under the`generate_data` function in `util_functs.py`, but I will go through the indiviidual steps of the data generation anyway.

The first task is to create our input clips from which mel-spectrograms are produced. The clips are 6 seconds in length, a number I found to be reasonably short enough for song prediction. This code snippet illustrates the clip generation process:

```python
from pydub import AudioSegment

song = AudioSegment.from_wav(song_path) #extract audio

STRIDE = stride * 1000 #pydub works in ms
CLIP_LENGTH = clip_length * 1000 #pydub works in ms
num_clips = math.floor((len(song) - CLIP_LENGTH) / STRIDE) + 1

for t in range(num_clips):
    clip = song[STRIDE*t:CLIP_LENGTH + STRIDE*t]
    clip.export(output_path + "clip" + str(t) + ".wav", format="wav")
```
This was taken from the `audio_splice` function in `util_functs.py`. Essentially, we load the song into memory, the length of which is represented in milliseconds (it's a pydub thing) and slide along it, extracting 6-esecond snippets at each step and saving them. 

If you have knowledge of how CNNs work, you know that once the "filter" or "kernel" has performed the convolution operation on area of the input image, it steps along to the next area of the image with a certain stride. The output size of the convolution operation, (assuming no [padding](https://medium.com/@ayeshmanthaperera/what-is-padding-in-cnns-71b21fb0dd7)) is defined by $$n-f/ s + 1$$, where `f` is the filter size. Although these are audio clips and not images, the same rule can be applied to a 1D temporal data stream (ie. an audio clip). We just take `n` to be the length of the raw audio song file, `f` to be the desired clip length, and `s` to be the step size when sliding along the audio file. 


## Running the CNN
## Closing Thoughts
