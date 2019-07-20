# HumGuess - Multi-class classification ML project

In this repository, you will find helpful notebooks detailing the HumGuess project - where a machine learning approach is used to
predict what song someone is singing/humming. The approach used was inspired by work done by John Hartquist on an experimental [fast.ai audio classification module](https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89), and by a CNN approach to [classifiying MNIST digits](https://medium.com/x8-the-ai-community/audio-classification-using-cnn-coding-example-f9cbd272269e). To be able to understand the notebooks fully, please follow this README carefully.

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

graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;


## Running the CNN
## Closing Thoughts
