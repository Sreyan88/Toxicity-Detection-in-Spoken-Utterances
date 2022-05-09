#Importing Libraries
import re
import os
import time
import sys
import json
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import librosa
import pandas as pd
from functools import reduce
from sklearn.metrics import f1_score

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertConfig

from utils import create_processor, prepare_example
from mmi_module_only_audio import MMI_Model_Single

import warnings
warnings.filterwarnings("ignore")


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
audio_processor = create_processor("facebook/wav2vec2-base")

vocabulary_chars_str = "".join(t for t in audio_processor.tokenizer.get_vocab().keys() if len(t) == 1)
vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
		f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
		flags=re.IGNORECASE if audio_processor.tokenizer.do_lower_case else 0,
	)

def evaluate_metrics(pred_label, true_label):
	pred_label = np.array(pred_label)
	true_label = np.array(true_label)
	macro=f1_score(true_label, pred_label, average='macro')
	micro=f1_score(true_label, pred_label, average='micro')
	key_metric, report_metric = macro, {'Macro':macro,'Micro':micro}

	return key_metric, report_metric

class IEMOCAPDataset(object):
	def __init__(self, config, data_list):
		self.data_list = data_list
		self.unit_length = int(8 * 16000)
		self.audio_length = config['acoustic']['audio_length']
		self.feature_name = config['acoustic']['feature_name']
		self.feature_dim = config['acoustic']['embedding_dim']

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, index):
		audio_path, bert_text, label = self.data_list[index]
		audio_name = os.path.basename(audio_path)

		#------------- extract the audio features -------------#
		wave,sr = librosa.core.load(audio_path + ".wav", sr=16000)
		audio_length = len(wave)

		#------------- wrap up all the output info the dict format -------------#
		return {'audio_input':wave,'text_input':bert_text,'audio_length':audio_length,
				'label':label,'audio_name':audio_name}


def collate(sample_list):
	
	batch_audio = [x['audio_input'] for x in sample_list]
	batch_bert_text = [x['text_input'] for x in sample_list]

	#----------------tokenize and pad the audio----------------------#

	batch_audio = audio_processor(batch_audio, sampling_rate=16000).input_values

	batch_audio = [{"input_values": audio} for audio in batch_audio]
	batch_audio = audio_processor.pad(
			batch_audio,
			padding=True,
			return_tensors="pt",
		)

	#----------------tokenize and pad the extras----------------------#
	audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])


	batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
	batch_name = [x['audio_name'] for x in sample_list]

	return (batch_audio,audio_length),(batch_label)


def run(args, config, train_data, valid_data, test_data,trigger_data):

	############################ PARAMETER SETTING ##########################
	num_workers = 1
	batch_size = 1
	epochs = args.epochs
	learning_rate = 0.00001 #5e-4 #0.00001-best with sigle loss wav2vec2 #5e-4 #5*1e-5
	accum_iter = 8
	stats_file_valid = open("best_valid" + '_' + 'stats.txt', 'a', buffering=1)
	stats_file_test = open("best_test" + '_' + 'stats.txt', 'a', buffering=1)
	stats_file_testt = open("best_testt" + '_' + 'stats.txt', 'a', buffering=1)

	############################## PREPARE DATASET ##########################
	train_dataset = IEMOCAPDataset(config, train_data)
	train_loader = torch.utils.data.DataLoader(
		dataset = train_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x),
		shuffle = True, num_workers = num_workers
	)
	valid_dataset = IEMOCAPDataset(config, valid_data)
	valid_loader = torch.utils.data.DataLoader(
		dataset = valid_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x),
		shuffle = False, num_workers = num_workers
	)
	test_dataset = IEMOCAPDataset(config, test_data)
	test_loader = torch.utils.data.DataLoader(
		dataset = test_dataset, batch_size = 1, collate_fn=lambda x: collate(x),
		shuffle = False, num_workers = num_workers
	)
	trigger_dataset = IEMOCAPDataset(config, trigger_data)
	trigger_loader = torch.utils.data.DataLoader(
		dataset = trigger_dataset, batch_size = 1, collate_fn=lambda x: collate(x),
		shuffle = False, num_workers = num_workers
	)

	########################### CREATE MODEL #################################
	config_mmi = BertConfig('config.json')
	model = MMI_Model_Single(config_mmi,len(audio_processor.tokenizer),2)
	model.cuda()
	#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name)
	
	########################### TRAINING #####################################
	count, best_metric, save_metric, best_epoch = 0, -np.inf, None, 0

	for epoch in range(epochs):
		epoch_train_loss = []
		epoch_train_cls_loss = []
		epoch_train_ctc_loss = []
		
		model.train()
		start_time = time.time()
		batch_idx = 0
		time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
		progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
		for audio_input, label_input in progress:
			
			acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
			emotion_labels = label_input.cuda()

			
			#model.zero_grad()
			loss, _, cls_loss, ctc_loss = model(0, 0, 0, 0, acoustic_input, acoustic_length, 0, emotion_labels, 0)

			loss = loss/accum_iter
			loss.backward()

			if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
				epoch_train_loss.append(loss)
				epoch_train_cls_loss.append(cls_loss)
				epoch_train_ctc_loss.append(ctc_loss)
				optimizer.step()
				optimizer.zero_grad()

			batch_idx += 1
			count += 1
			acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
			cls_loss = torch.mean(torch.tensor(epoch_train_cls_loss)).cpu().detach().numpy()
			ctc_loss = torch.mean(torch.tensor(epoch_train_ctc_loss)).cpu().detach().numpy()
			progress.set_description("Epoch {:0>3d} - Loss {:.4f} - CLS_Loss {:.4f} - CTC_Loss {:.4f}".format(epoch, acc_train_loss, cls_loss, ctc_loss))



		model.eval()
		pred_y, true_y = [], []
		with torch.no_grad():
			time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
			for audio_input, label_input in tqdm(valid_loader):
				acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
				emotion_labels = label_input.cuda()

				true_y.extend(list(emotion_labels.cpu().numpy()))

				_, logits, cls_loss, ctc_loss = model(0, 0, 0, 0, acoustic_input, acoustic_length, 0, emotion_labels, 0)
				
				prediction = torch.argmax(logits, axis=1)
				label_outputs = prediction.cpu().detach().numpy().astype(int)
				
				pred_y.extend(list(label_outputs))

		key_metric, report_metric = evaluate_metrics(pred_y, true_y)

		epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print('Valid Metric: {} - Train Loss: {:.3f}'.format(
			' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()]),
			epoch_train_loss))
		stats = dict(epoch=epoch, key_accuracy = key_metric, report_accuracy = report_metric)
		print(json.dumps(stats), file= stats_file_valid)

		if key_metric > best_metric:
			torch.save({'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},"DeToxy/FG_CME/checkpoints/MELD/" + "best" + '_' + "model.pt")
			#best_metric, best_epoch = key_metric, epoch
			print('Better Metric found on dev, calculate performance on Test')
			pred_y, true_y = [], []
			with torch.no_grad():
				time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
				for audio_input, label_input in test_loader:
					acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
					emotion_labels = label_input.cuda()

					true_y.extend(list(emotion_labels.cpu().numpy()))

					_, logits, _, _ = model(0, 0, 0, 0, acoustic_input, acoustic_length, 0, emotion_labels, 0)
				

					prediction = torch.argmax(logits, axis=1)
					label_outputs = prediction.cpu().detach().numpy().astype(int)

					pred_y.extend(list(label_outputs))        
			
			_, save_metric = evaluate_metrics(pred_y, true_y)
			stats = dict(epoch=epoch, report_accuracy = save_metric)
			print(json.dumps(stats), file= stats_file_test)
			print("Test Metric: {}".format(
				' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])
			))

		if key_metric > best_metric:
			#torch.save({'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},"DeToxy/FG_CME/checkpoints/MELD/" + "best" + '_' + "model.pt")
			best_metric, best_epoch = key_metric, epoch
			print('Better Metric found on dev, calculate performance on Trigger')
			pred_y, true_y = [], []
			with torch.no_grad():
				time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
				for audio_input, label_input in trigger_loader:
					acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
					emotion_labels = label_input.cuda()

					true_y.extend(list(emotion_labels.cpu().numpy()))

					_, logits, _, _ = model(0, 0, 0, 0, acoustic_input, acoustic_length, 0, emotion_labels, 0)
				

					prediction = torch.argmax(logits, axis=1)
					label_outputs = prediction.cpu().detach().numpy().astype(int)

					pred_y.extend(list(label_outputs))        
			
			_, save_metric = evaluate_metrics(pred_y, true_y)
			stats = dict(epoch=epoch, report_accuracy = save_metric)
			print(json.dumps(stats), file= stats_file_testt)
			print("Trigger Term Metric: {}".format(
				' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])
			))

	print("End. Best epoch {:03d}: {}".format(best_epoch, ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])))
	return save_metric


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="config/iemocap-ours.yaml", help='configuration file path')
	parser.add_argument("--epochs", type=int, default=50, help="training epoches")
	parser.add_argument("--csv_path", type=str, default="../data/", help="path of csv")
	parser.add_argument("--save_path", type=str, default="./", help="report or ckpt save path")
	parser.add_argument("--data_path", type=str, default="../DeToxy/Input/", help="Audio Files Path")
	parser.add_argument("--data_path_bert", type=str, default="numpy_bert/", help="report or ckpt save path")
	
	args = parser.parse_args()

	config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
	report_result = []

	df_emotion_train=pd.read_csv(args.csv_path+"train.csv",encoding='latin-1')
	df_emotion_train.rename(columns={'text':'Utterance','label2a':'Emotion'},inplace=True)
	
	df_emotion_valid=pd.read_csv(args.csv_path+"valid.csv",encoding='latin-1')
	df_emotion_valid.rename(columns={'text':'Utterance','label2a':'Emotion'},inplace=True)

	df_emotion_test=pd.read_csv(args.csv_path+"test.csv",encoding='latin-1')
	df_emotion_test.rename(columns={'text':'Utterance','label2a':'Emotion'},inplace=True)

	df_emotion_trigger=pd.read_csv(args.csv_path+"trigger_test.csv",encoding='latin-1')
	df_emotion_trigger.rename(columns={'text':'Utterance','label2a':'Emotion'},inplace=True)

	train_data = []
	valid_data = []
	test_data = []
	trigger_data = []

	for row in df_emotion_train.itertuples():
		file_name = args.data_path + "train_splits_audio/" + str(row.FileName)
		train_data.append((file_name,row.Utterance,row.Emotion))

	for row in df_emotion_valid.itertuples():
		file_name = args.data_path + "dev_splits_audio/" + str(row.FileName)
		valid_data.append((file_name,row.Utterance,row.Emotion))
	
	for row in df_emotion_test.itertuples():
		file_name = args.data_path + "test_splits_audio/" + str(row.FileName)
		test_data.append((file_name,row.Utterance,row.Emotion))

	for row in df_emotion_trigger.itertuples():
		file_name = args.data_path + "trigger_test_audio/" + str(row.FileName)
		trigger_data.append((file_name,row.Utterance,row.Emotion))

	report_metric = run(args, config, train_data, valid_data, test_data, trigger_data)

	report_result.append(report_metric)

	os.makedirs(args.save_path, exist_ok=True)
	pickle.dump(report_result, open(os.path.join(args.save_path, 'metric_report.pkl'),'wb'))
