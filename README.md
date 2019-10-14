# HumGuess - Multi-class classification

In this repository, you will find helpful notebooks detailing the HumGuess project - where a machine learning approach is used to
predict what song someone is singing/humming. This projecy was inspired by work from John Hartquist on an experimental [fast.ai audio classification module](https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89), and by a CNN approach to [classifiying MNIST digits](https://medium.com/x8-the-ai-community/audio-classification-using-cnn-coding-example-f9cbd272269e). For a more techincal insight into the methods used, please consult these resources.

## Table of Contents

1. [Overview](https://github.com/izwauld/HumGuess#overview)
2. [Setup](https://github.com/izwauld/HumGuess#setup)
3. [Data Generation](https://github.com/izwauld/HumGuess#data-generation)
4. [Running the CNN](https://github.com/izwauld/HumGuess#running-the-cnn) 
5. [Closing Thoughts](https://github.com/izwauld/HumGuess#closing-thoughts)

## Overview

Have you ever been listening to music, and come across a song that you really like the sound of, but you don't know the name of it? You might pick up on some of the lyrics of the song, but it isn't enough to be able to know what the song name is. If the song is popular enough, you may be able to type in the lyrics into some search engine and it will fetch the song for you, but that isn't always a reliable way to get the name of the song. If you play the song to Shazam or Google, it might be able to recgonise it for you. But now it's too late - the song has passed. All you have to go off of is the beat that is stuck in your head.

So, let's harness that!

The purpose of this project is to build a machine learning system that will be able to predict what song you are singing or humming. For this project, I framed the problem as a supervised, multi-label classification problem. Basically, The ML model is trained to predict one of five songs, where you input an audio clip into the ML algorithm, and it spits out a probability that the clip refers to one of the five songs.

The model is trained on three accents currently, just to demonstrate that this sort of approach can work (relatively) well. Ideally you would like to train on many voices to make the model more robust, but in fact, the guiding intuition is that the model doesn't really need to be trained on that many voices, just a more varied selection. This is because of the song choice: the songs chosen for this project are easily recognisable (like Michael Jackson - Thriller), so intuitively the frequency patterns of the audio clips should be similar regardless of the singer's accent. This means that you don't necessarily have to train on million's of voices! 

One major drawback to this approach is scalability: there are millions of songs out there, and more being released every day. This means that in order for this approach to be super robust, the model would have to be able to correctly identify millions of songs, but even if you got that to work, it would need to be retrained every time to account for new songs being released. So clearly, that isn't feasible!

With the high-level introduction out the way, we move onto the setup, where we introduce the packages and dependencies that are needed.


## Setup

The following tools/dependencies were used in the project:
* [Python](https://www.python.org/) - version 3.7.3
* [Tensorflow](https://www.tensorflow.org/) - high-level machine learning framework, version 1.14.0
* (recommended) [Anaconda](https://www.anaconda.com/) - Data science distribution (comes with Jupyter notebook), version 4.7.5
* [pydub](https://pypi.org/project/pydub/) - audio package for Python, version 0.23.1
* [librosa](https://librosa.github.io/librosa/) - audio package for Python, version 0.6.3


## Data Generation

The first step is to create the input data, namely the mel-spectrograms that will be fed into the CNN. All operations are sheltered under the`generate_data` function in `util_functs.py`, but I will go through the indiviidual steps of the data generation anyway.


The first task is to create our **input clips** from which mel-spectrograms are produced. The clips are 6 seconds in length (a number I found to be reasonably short enough for song prediction, but it is tunable). This code snippet illustrates the clip generation process:

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

If you have knowledge of how CNNs work, you know that once the "filter" or "kernel" has performed the convolution operation on an area of the input image, it steps along to the next area with a certain stride. The output size of the convolution operation, (assuming no [padding](https://medium.com/@ayeshmanthaperera/what-is-padding-in-cnns-71b21fb0dd7)), is defined as `|(n - f) / s| + 1` | refers to the "floor" operation here), where `f` is the filter size. Although these are audio clips and not images, the same rule can be applied to a 1D temporal data stream (ie. an audio clip). We just take `n` to be the length of the raw audio song file, `f` to be the desired clip length, and `s` to be the step size when sliding along the audio file. 


Next, we generate the **mel-spectrograms**. These are plots of the most common frequencies from the input audio signals, mapped to a logarithmic scale (or "mel" scale) since these humans process noise on these scales. The code snippet below illustrates how the plots are generated:

```python
    
   import librosa, librosa.display
   from os import splitext
   my_dpi=96
   
   fname = splitext(clip_path)[0]
   label = clip_path[0]
        
   clip, sample_rate = librosa.load(clip_path, sr=None)
   mel_spec = librosa.feature.melspectrogram(clip, n_fft=n_fft, hop_length=n_hop,
                                          n_mels=n_mels, sr=sample_rate, power=1.0, 
                                          fmin=fmin, fmax=fmax)
   mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        
   plt.figure(figsize=(50/my_dpi, 34/my_dpi), dpi=my_dpi) #(34,50)
   librosa.display.specshow(mel_spec_db,
                                 sr=sample_rate, hop_length=n_hop,
                                 fmin=fmin, fmax=fmax)
   plt.savefig(melspec_output_path)
   plt.close()
```

This has been adapted from the more general-purpose function `generate_mel_specs` in `util_functs.py` to show how a single image would be created and saved. The `clip_path` and `melspec_output_path` variables should be self-explanatory.


What follows are more preprocessing steps (storing a % of images away as test data, creating pandas dataframe containing filename-label pairs, then creating a .h5 file consisting of image-label data). You can find the details of these steps in the `GenerateData.ipynb` notebook.

That concludes the discussion on how the input data is generated. Now, we showcase the structure of the CNN model.


## Running the CNN

Here, [Keras](https://keras.io/) was used to build the model. The architecture of the model is shown here (see `cnn_architecture.py` also):

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), 
                 activation='relu', input_shape=X_train[1].shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.20))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.50))
model.add(Dense(5, activation='softmax'))
```

This architecture was taken from the 'classifiying MNIST digits' reference given above. The only difference is the addition of another pooling layer after the second Conv2D layer. This architecture has proved to work well for this task (as you will see). The final softmax layer should come as no surprise - the final output layer should be a vector of proabilities assigned to each label, which is what the softmax vector is. The two dropout layers help to prevent the model from becoming too reliant on any single input features (having a "regularizing" effect). 

The model is then compiled and run as follows:

```python
model.compile(loss = "categorical_crossentropy", optimizer = 'adam',
              metrics=['accuracy'])
model.fit(X_train, y_train_oh, batch_size=16, epochs=50, verbose=1, validation_split=0.20)
```
where `X_train` contains the image arrays and `y_train_oh` is the [one-hot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) of the labels. `validation_split=0.20` means that 20% of the data is used as the validation set.

After the model runs, we save the output model and make a prediction on the test set (see `TestModel.ipynb`):

```python
model = load_model(model_path)

predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1) + 1
predictions = predictions.reshape(len(ms_df_test),1)

#Calculate the difference between the predictions and true labels, then find where the difference is 0 (correct predictions) 
diff_array = predictions - y_test
correct_preds = diff_array[diff_array == 0]

#Finally, calculate the accuracy of the model on the test data
acc_percent = (len(correct_preds) / len(diff_array)) * 100
print(acc_percent)
```


## Closing Thoughts

The model performed well, consistently performing ~80% when I tried it out. As mentioned earlier, the model could be more robust if trained on a more varied dataset (female voice, richer accent variety, etc.) so that could be a cool avenue to explore. Overall, this was a fun little project to work on :-)
