# Piano music transcription using Convolutional Neural Networks

This is the final project of Stanford CS230: Deep learning

## Introduction

This project tries to solve piano music transcription problem using convolutional neural networks (CNN). The input to our system is piano music in .WAV format. We then apply constant Q transform (CQT) to the input so that the audio file is divided into multiple frames. Each frame has a ground truth label, thus we can use our neural networks to predict the pitch within each frame. The output is then processed to a MIDI (Musical Instrument Digital Interface) file, which can be directly converted to music scores.

## Dataset
The dataset we use for this project is MAPS (MIDI Aligned Piano Sounds) dataset. Here is the [link](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) to the dataset.

## Source Code
- Preprocessing and evaluation scripts: forked from [this repo](https://github.com/diegomorin8/Deep-Neural-Networks-for-Piano-Music-Transcription)
- **model/train.py** defines models and training process
- **model/train_load.py** loads a previous checkpoint
- **script/npy2midi.py** coverts output numpy arrays to MIDI files.
- **script/run.sh** integrates data preprocessing, model prediction and MIDI conversion together.
 
 ## Prediction Result
 Here is a [Youtube video](https://youtu.be/u-_W-XH7EJg) that presents the prediction result.
