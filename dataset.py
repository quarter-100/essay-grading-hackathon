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
            "신이 난" : 0, "안도" : 1, "흥분" : 2, "기쁨" : 3, "만족스러운" : 4, "만족스러운" : 5, "자신하는" : 6, "편안한" : 7, "감사하는" : 8, "신뢰하는" : 9, "느긋" : 10,
            "우울한" : 10, "눈물이 나는" : 11, "낙담한" : 12, "마비된" : 13, "염세적인" : 14, "환멸을 느끼는" : 15, "비통한" : 16, "실망한" : 17, "슬픔" : 18, "후회되는" : 19, 
            "스트레스 받는" : 20, "취약한" : 21, "당혹스러운" : 22, "두려운" : 23, "회의적인" : 24, "걱정스러운" : 25, "조심스러운" : 26, "불안" : 27, "혼란스러운" : 28, "초조한" : 29, 
            "당황" : 30, "남의 시선을 의식하는" : 31, "죄책감의" : 32, "한심한" : 33, "혐오스러운" : 34, "부끄러운" : 35, "열등감" : 36, "외로운" : 37, "고립된" : 38, 
            "구역질 나는" : 39, "좌절한" : 40, "분노" : 41, "안달하는" : 42, "노여워하는" : 43, "짜증내는" : 44, "방어적인" : 45, "툴툴대는" : 46, "성가신" : 47, "악의적인" : 48, 
            "배신당한" : 49, "충격 받은" : 50, "상처" : 51, "억울한" : 52, "버려진" : 53, "괴로워하는" : 54, "질투하는" : 55, "가난한, 불우한" : 56, "희생된" : 57
        }

        for val in label:
            #num_label.append(dict_label_to_num[val])
            num_label.append(val)
        
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
        item['labels'] = torch.tensor(self.labels[idx], dtype = torch.float32)

        return item
    
    def __len__(self):
        return len(self.labels)
