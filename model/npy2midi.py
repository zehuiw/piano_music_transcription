import numpy as np
import sys
import os
import librosa
from scipy.io import savemat
from midiutil import MIDIFile

matY = np.load('../Preprocessing/dev/0dev_y.npy')
file = open( 'sample.txt' , "w")
MyMIDI = MIDIFile(1)


track    = 0
channel  = 0
time     = 0    # In beats
duration = 1    # In beats
tempo    = 60   # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI.addTempo(track, time, tempo)

frame_width = 512.0
sample_frequence = 44100.0

pitch = 20

for col in np.transpose(matY):
	startTime = 0.0
	endTime = 0.0
	pitch += 1
	status = False
	for idx, i in enumerate(col):
		if i > 0.5 and status == False:
			startTime = idx * frame_width / sample_frequence
			status = True
		if i < 0.1 and status == True:
			endTime = idx * frame_width / sample_frequence
			status = False
			if endTime - startTime > 0.02:
				MyMIDI.addNote(track, channel, pitch, startTime, endTime - startTime, volume)
				file.write("%f, %f, %d\n" % (startTime, endTime, pitch))
	if status:
		if (len(col) * frame_width - 0.5) / sample_frequence - startTime > 0.02:
			MyMIDI.addNote(track, channel, pitch, startTime, (len(col) * frame_width - 0.5) / sample_frequence - startTime, volume)
			file.write("%f, %f, %d\n" % (startTime, (len(col) * frame_width - 0.5) / sample_frequence, pitch))
file.close()
with open("sample.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)