# DeToxy: A Large-Scale Multimodal Dataset for Toxicity Classification in Spoken Utterances

## Introduction

DeToxy is a large scale multimodal dataset created by manually annotating toxic labels onto spoken utterances that allows content moderation researchers to encompass audio modality onto their work as well. DeToxy is sourced from various openly available speech databases and consists of over 2 million utterances. The datasets that have been used for creating DeToxy are CMU-MOSEI, CMU-MOSI, Common Voice, IEMOCAP, LJ Speech, MELD, MSP-Improv, MSP-Podcast, Social-IQ, SwitchBoard and VCTK. We also provide DeToxy-B, a
balanced version of the dataset, curated from the original larger version taking into consideration auxiliary factors like trigger terms and utterance sentiment labels. 

## Dataset Statistics
<p align="center">
<img src='https://user-images.githubusercontent.com/33536225/167460726-ea49b960-e67d-4f7b-8799-911b5d542266.png' height = 500 width = 750>
</p>

## Paper
The paper containing the detailed explanation of the dataset can be found here - https://arxiv.org/pdf/2110.07592.pdf

## Objective
Social network platforms are generally meant to share positive, constructive, and insightful content. However, in recent times, people often get exposed to objectionable content like threats, identity attacks, hate speech, insults, obscene texts, offensive remarks, or bullying. With the rise of different forms of content available online beyond just written text, i.e., audio and video, it is crucial that we device efficient content moderation systems for these forms of shared media. However, most prior
work in literature and available datasets focus primarily on the modality of conversational text, with other modalities of conversation ignored at large. Thus, to alleviate this problem, in this paper, we propose a new dataset DeToxy, for the relatively new and unexplored Spoken Language Processing (SLP) task of toxicity classification in spoken utterances, which remains a crucial problem to solve for interactive intelligent systems, with broad applications in the field of content moderation in online audio/video content, gaming, customer service, etc.

## Download the data

Please visit - "some zenodo link" to download the raw data. Data are stored in .wav format and can be found in XXX.tar.gz files. Annotations can be found in the data <a href="https://github.com/SamdenLepcha/Toxicity-Detection-in-Spoken-Utterances/tree/main/data">folder</a>.

## Description of the .csv files

| Column Name | Description |
| --- | --- |
| Dataset | This column gives the name of the dataset. |
| FileName | The Filename of each audio file from datasets. |
| text | Individual utterances from each audio file as a string. |
| label2a | The label (toxic: 1, non-toxic: 0) annotated for each utterance. |
| Starting | The starting time of the utterance in the given audio file in seconds (only used to segment data).  |
| Ending | The ending time of the utterance in the given audio file in seconds (only used to segment data). |

### The files

/data/metadata.csv - contains notes for the preparation of each dataset. <br>
/data/test.csv  - contains the utterances in the test set along with Toxicity Labels and Starting/Ending Time. <br>
/data/train.csv -  contains the utterances in the training set along with Toxicity Labels and Starting/Ending Time.<br>
/data/trigger_test.csv - contains the utterances in the trigger term test set along with Toxicity Labels and Starting/Ending Time. <br>
/data/valid.csv - contains the utterances in the validation set along with Toxicity Labels and Starting/Ending Time. <br>

## Setup Instructions

<ol>
<li>Clone the entire repository into your local machine.</li>
<li>Search and download online all the open source datasets mentioned in the introduction and place them all in a new folder in this directory.</li>
<li> Open Anaconda Command Prompt and Setup a new environment</li><br>
   
```
 C:\> conda create -n DeToxy pip python=3.6
```
<li> Activate the environment and upgrade pip. </li><br>
  
```
C:\> activate DeToxy
(DeToxy) C:\>python -m pip install --upgrade pip
```
<li> All other requirements can be installed using requirements.txt</li><br>

```
 (DeToxy) C:\>pip install -r requirements.txt
```
  
<li> Using the Audio (Final DeToxy-B Audio Prep.ipynb) and Transcript (Final DeToxy-B Transcript Prep.ipynb) Jupyter Notebooks present in the data prep folder to prepare the datasets. The Gold Transcripts are already present in the data folder and is not required to run. </li> 
  
```
 (DeToxy) C:\> jupyter notebook
```
  
<li> The codes present in the two_step_approach folder is used to run the two step experiments present in the paper. First use the transcribe.py code is used to generate the transcripts for all the audio files using models pretrained on Librispeech, Common Voice and Switchboard. </li><br>
<p align="left">
<img src='https://user-images.githubusercontent.com/33536225/167469507-ac732b4c-ed96-4d78-80d2-335a3d42c536.png' height = 200 width = 650>
</p>
  
```
 (DeToxy) C:\Toxicity-Detection-in-Spoken-Utterances\e2e\> python transcribe.py
```
  
<li> After creating the transcripts for each dataset and pretrained model use the Civil_Comments Jupyter Notebook to predict and evaluate using the model trained on publicly available toxic dataset(text). </li>
<li> To perform the DB(DeToxy-B) two step approach find the main.py file present in the two_step_approach/DB folder. Make changes to the default paths accordingly inside the parse function. Also make changes to the csv file names preset in the main function. Change the name of the Stats File according to the transcripts that is being evaluated. The training dataset will remain the same dataset for all the other transcripts. Then run the main.py file. </li> <br>
  
```
 (DeToxy) C:\Toxicity-Detection-in-Spoken-Utterances\two_step_approach\DB\> python main.py
```
  
<li> For the End to End approach, first select the model that you want to train on. This change has to be done in line 1559 under the MMI_Model_Single class in mmi_module_only_audio.py. Change it accordingly in run_iemocap-ours-meld.py under line 32 and 33. In the main function the paths for files can be changed according to one's needs. Then run the run_iemocap script using the line below.</li>
  
```
 (DeToxy) C:\Toxicity-Detection-in-Spoken-Utterances\e2e\> python run_iemocap-ours-meld.py 
```

</ol>

## Citation
Please cite the following paper if you find this dataset useful in your research: <br>
S. Ghosh, S. Lepcha, Sakshi, R. R. Shah and S. Umesh. DeToxy: A Large-Scale Multimodal Dataset for Toxicity Classification in Spoken Utterances. arXiv preprint arxiv.2110.07592 (2022).