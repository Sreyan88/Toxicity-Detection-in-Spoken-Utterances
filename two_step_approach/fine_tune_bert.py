#Importing Libraries
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import torch
cuda_available = torch.cuda.is_available()

#To Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#Importing File
data=pd.read_csv("../Transcripts/Final.csv")

#Subsetting Train and Eval Datasets
train_df=data[data['split']==0]
train_df=train_df[['text','label2a']]
train_df.rename(columns={'label2a':'labels'},inplace=True)
train_df.reset_index(drop=True,inplace=True)

eval_df=data[data['split']==1]
eval_df=eval_df[['text','label2a']]
eval_df.rename(columns={'label2a':'labels'},inplace=True)
eval_df.reset_index(drop=True,inplace=True)

#Optional model configuration
model_args = ClassificationArgs()

#Setting Model Parameters
model_args.labels_list=[1,0]
model_args.num_train_epochs=10
model_args.use_early_stopping = True
model_args.early_stopping_consider_epochs = 4

#Create a Classification Model
model = ClassificationModel(
    "bert", "bert-base-uncased",
    use_cuda=True,
    args=model_args,
    num_labels = 2
)

# Train and save the best model
model.train_model(train_df,eval_df=eval_df,output_dir='model_output/',best_model_dir='outputs/best_model',eval_during_training = True)


