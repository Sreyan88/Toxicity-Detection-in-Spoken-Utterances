#Importing libraries
import librosa
import os
import pandas as pd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import warnings
warnings.filterwarnings("ignore")

#Setting source path
source='Dataset/Input/test_splits_audio/'
destination='Dataset/Transcribed_Transcripts/'

#Getting all file names
files = os.listdir(source)

#For GPU use if available
torch.cuda.set_device(1)
device = "cuda:1" if torch.cuda.is_available() else "cpu"

#Importing the pretrained models
#libri_speech='facebook/wav2vec2-large-960h'
#common_voice='patrickvonplaten/wav2vec2-large-xls-r-300m-common_voice-tr-ft'
switchboard='facebook/wav2vec2-large-robust-ft-swbd-300h'

processor = Wav2Vec2Processor.from_pretrained(switchboard)
model = Wav2Vec2ForCTC.from_pretrained(switchboard)
model = model.to(device)

#Creating a new txt file with transcripts
myfile = open(destination+"SwitchBoard_Test_transcripts.txt","w")

for i in range(len(files)):
    
    print("Wav File "+str(i+1)+"/"+str(len(files)))
    
    #load any audio file of your choice
    speech, rate = librosa.load(source+files[i],sr=16000)
    input_values = processor(speech, return_tensors = 'pt').input_values.to(device)

    #Store logits (non-normalized predictions)
    logits = model(input_values).logits

    #Store predicted id's
    predicted_ids = torch.argmax(logits, dim =-1)

    #decode the audio to generate text
    transcriptions = processor.decode(predicted_ids[0])
    
    myfile.write(files[i]+";"+str(transcriptions))
    myfile.write('\n')
    
    del input_values, logits, predicted_ids,transcriptions,speech,rate

myfile.close()
