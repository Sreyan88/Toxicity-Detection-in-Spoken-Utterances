#Importing all Libraries
import pandas as pd
import numpy as np
import librosa
import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

#Initialising Paths

source='../Testing/Input/test/'

destination='../Testing/Output/'

#Calling Transcribing Function
get_transcripts(source,destination)

#Importing Saved Transcripts

data=pd.read_csv(destination+"Transcripts.txt",sep=';')

data.columns = ["text", "labels"]

#Running Locally Saved Toxicity Model

cuda_available = torch.cuda.is_available()

model = ClassificationModel(
    "bert", "outputs/best_model", use_cuda=cuda_available
)

#Making predictions
predictions=[]

for i in range(data.shape[0]):

	temp_predictions = model.predict([data['text'].iloc[i]])

	predictions.append(temp_predictions)

data['Predictions']=predictions

#Evaluating Results

print(classification_report(data['labels'], data['Predictions']))

