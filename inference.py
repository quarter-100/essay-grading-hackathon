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
    parser.add_argument('--test_path', type=str, default='/opt/ml/sentiment_analysis/data/test.csv',
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
        "신이 난" : 0, "안도" : 1, "흥분" : 2, "기쁨" : 3, "만족스러운" : 4, "만족스러운" : 5, "자신하는" : 6, "편안한" : 7, "감사하는" : 8, "신뢰하는" : 9, "느긋" : 10,
        "우울한" : 10, "눈물이 나는" : 11, "낙담한" : 12, "마비된" : 13, "염세적인" : 14, "환멸을 느끼는" : 15, "비통한" : 16, "실망한" : 17, "슬픔" : 18, "후회되는" : 19, 
        "스트레스 받는" : 20, "취약한" : 21, "당혹스러운" : 22, "두려운" : 23, "회의적인" : 24, "걱정스러운" : 25, "조심스러운" : 26, "불안" : 27, "혼란스러운" : 28, "초조한" : 29, 
        "당황" : 30, "남의 시선을 의식하는" : 31, "죄책감의" : 32, "한심한" : 33, "혐오스러운" : 34, "부끄러운" : 35, "열등감" : 36, "외로운" : 37, "고립된" : 38, 
        "구역질 나는" : 39, "좌절한" : 40, "분노" : 41, "안달하는" : 42, "노여워하는" : 43, "짜증내는" : 44, "방어적인" : 45, "툴툴대는" : 46, "성가신" : 47, "악의적인" : 48, 
        "배신당한" : 49, "충격 받은" : 50, "상처" : 51, "억울한" : 52, "버려진" : 53, "괴로워하는" : 54, "질투하는" : 55, "가난한, 불우한" : 56, "희생된" : 57
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
        model_config.num_labels = 58

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