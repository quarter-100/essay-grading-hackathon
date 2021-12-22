import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
import pickle

class Preprocess:
    def __init__(self, CSV_PATH):
        self.data= self.load_data(CSV_PATH)

    def load_data(self, path):
        
        df = pd.read_csv(path)
        
        return df
    
    def label_to_num(self, label):
        num_label = [] # 숫자로 된 label 담을 변수

        dict_label_to_num = {
            "D" : 0, "C" : 1, "B" : 2, "A" : 3
        }

        for val in label:
            num_label.append(dict_label_to_num[val])
        
        return num_label

    def tokenized_dataset(self, data, tokenizer):

        tokenized_sentence= tokenizer(
            list(data['paragraph_txt']), # list나 string type으로 보내줘야 함 !
            return_tensors= "pt", # pytorch type
            padding= True, # 문장의 길이가 짧다면 padding
            truncation= True, # 문장 자르기
            max_length= 256, # 토큰 최대 길이...
            add_special_tokens= True, # special token 추가
            return_token_type_ids= False # roberta의 경우.. token_type_ids가 안들어감 ! 
        )    

        return tokenized_sentence

"""Train, Test Dataset"""
class Dataset:
    def __init__(self, data, labels): # data : dict, label : list느낌..
        self.data= data
        self.labels= labels
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item
    
    def __len__(self):
        return len(self.labels)
