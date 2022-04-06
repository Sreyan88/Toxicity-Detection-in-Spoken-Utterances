import textgrid
import torch
from torch import nn
import re

from transformers import (Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor)

def parse_Interval(IntervalObject):
    start_time = ""
    end_time = ""
    P_name = ""

    ind = 0
    str_interval = str(IntervalObject)
    for ele in str_interval:
        if ele == "(":
            ind = 1
        if ele == " " and ind == 1:
            ind = 2
        if ele == "," and ind == 2:
            ind = 3
        if ele == " " and ind == 3:
            ind = 4

        if ind == 1:
            if ele != "(" and ele != ",":
                start_time = start_time + ele
        if ind == 2:
            end_time = end_time + ele
        if ind == 4:
            if ele != " " and ele != ")":
                P_name = P_name + ele

    st = float(start_time)
    et = float(end_time)
    pn = P_name

    return (pn, st, et)


def parse_textgrid(filename):
    tg = textgrid.TextGrid.fromFile(filename)
    list_words = tg.getList("words")
    words_list = list_words[0]
    
    result = []
    for ele in words_list:
        d = parse_Interval(ele)
        result.append(d)
    return result

def create_processor(model_name_or_path,vocab_file= None):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path)

    if vocab_file:
        tokenizer = Wav2Vec2CTCTokenizer(
                vocab_file,
                do_lower_case=False,
                word_delimiter_token="|",
            )
    else:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_name_or_path,
                do_lower_case=False,
                word_delimiter_token="|",
            )
    return Wav2Vec2Processor(feature_extractor, tokenizer)





def prepare_example(text,vocabulary_text_cleaner):

    # Normalize and clean up text; order matters!
    try:
        text = " ".join(text.split())  # clean up whitespaces
    except:
        text = "NULL"
    updated_text = text
    updated_text = vocabulary_text_cleaner.sub("", updated_text)
    if updated_text != text:
        return re.sub(' +', ' ', updated_text).strip()
    else:
        return re.sub(' +', ' ', text).strip()

class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.
    It returns the mean and/or std of input tensor.
    Arguments
    ---------
    return_mean : True
         If True, the average pooling will be returned.
    return_std : True
         If True, the standard deviation will be returned.
    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor = sp_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    """

    def __init__(self, return_mean=True, return_std=True):
        super().__init__()

        # Small value for GaussNoise
        self.eps = 1e-5
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "both of statistics are equal to False \n"
                "consider enabling mean and/or std statistic pooling"
            )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                if self.return_mean:
                    mean.append(
                        torch.mean(x[snt_id, 0:actual_size, ...], dim=0)
                    )
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)

        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            gnoise = gnoise
            mean += gnoise
        if self.return_std:
            std = std + self.eps

        # Append mean and std of the batch
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self.return_std:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub("[\(\[].*?[\)\]]", '', text)

    # Replace '&amp;' with '&'
    text = re.sub(" +",' ', text).strip()

    return text