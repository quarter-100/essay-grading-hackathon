from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel

from dataset import *
#from model import *

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os

def get_test_config():
    parser= argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
    parser.add_argument('--kfold', type=int, default=1,
                        help='kfold (default: 1)')                   
    parser.add_argument('--model_path', type=str, default = './best_model', 
                        help='model load dir path (default : ./best_model)')
    parser.add_argument('--save_dir', type=str, default='./prediction',
                        help='submission save path')     
    parser.add_argument('--batch', type=int, default=64,
                        help='input batch size for test (default: 64') 
    parser.add_argument('--test_path', type=str, default='./test_with_para.csv',
                        help='test csv path') 

    args= parser.parse_args()

    return args

def inference(model, tokenized_data, device, args):
    dataloader= DataLoader(tokenized_data, batch_size= args.batch, shuffle= False)
    model.eval()
    output_pred, output_prob= [], []

    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs= model(
                input_ids= data['input_ids'].to(device),
                attention_mask= data['attention_mask'].to(device)
            )
        logits= outputs['logits']
        # prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits= logits.detach().cpu().numpy()
        prob= logits

        result= np.argmax(logits, axis= -1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis= 0).tolist()


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    dict_label_to_num = {
        "D" : 0, "C" : 1, "B" : 2, "A" : 3
    }
    dict_num_to_label = dict([(value, key) for key, value in dict_label_to_num.items()])
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label

def load_test_dataset(dataset_dir, tokenizer, args):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """

    preprocess= Preprocess(dataset_dir)
    
    test_dataset = preprocess.load_data(dataset_dir)
    #test_label = list(map(int,test_dataset['감정_소분류'].values))
    test_label = [0] * test_dataset.shape[0]
    tokenized_test = preprocess.tokenized_dataset(test_dataset, tokenizer)

    return tokenized_test, test_label

def to_nparray(s) :
    return np.array(list(map(float, s[1:-1].split(','))))

def main_inference(args):
    print('main inference start')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer= AutoTokenizer.from_pretrained(args.model)

    df_list= []
    for i in range(args.kfold):
        print(f'KFOLD : {i} inference start !')
        
        model_config =  AutoConfig.from_pretrained(args.model)
        model_config.num_labels = 4

        model =  AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config)
        #model= Model(args.model)
        #model.model.resize_token_embeddings(tokenizer.vocab_size + args.add_token)

        best_state_dict= torch.load(os.path.join(f'{args.model_path}', 'pytorch_model.bin'))
        model.load_state_dict(best_state_dict)
        model.to(device)
        
        test_dataset, test_label= load_test_dataset(args.test_path, tokenizer, args)
        testset= Dataset(test_dataset, test_label)

        pred_answer, output_prob= inference(model, testset, device, args)
        pred_answer= num_to_label(pred_answer)
        print(len(pred_answer), len(output_prob))

        output = pd.DataFrame({'pred_label':pred_answer,'probs':output_prob,})

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        output.to_csv(os.path.join(args.save_dir, f'submission.csv'), index= False)
        
        print(f'KFOLD : {i} inference fin !')

    print('FIN')


if __name__ == '__main__':
    args= get_test_config()
    main_inference(args)