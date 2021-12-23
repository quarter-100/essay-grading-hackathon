import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback
import argparse
import random
import argparse

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import wandb
from dataset import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_config():
    parser = argparse.ArgumentParser()


    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--model', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
    parser.add_argument('--train_path', type= str, default= './train.csv',
                        help='train csv path (default: ./train.csv')
    parser.add_argument('--save_dir', type=str, default = './best_model', 
                        help='model save dir path (default : ./best_model)')
    parser.add_argument('--wandb_path', type= str, default= 'roberta_classify_grade',
                        help='wandb graph, save_dir basic path (default: roberta_classify_grade')    
    parser.add_argument('--fold', type=int, default=5,
                        help='fold (default: 5)')
    parser.add_argument('--loss', type=str, default= 'LB',
                        help='LB: LabelSmoothing, CE: CrossEntropy')


    #hyperparameter
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--batch', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--gradient_accum', type=int, default=2,
                        help='gradient accumulation (default: 2)')
    parser.add_argument('--batch_valid', type=int, default=32,
                        help='input batch size for validing (default: 32)')
    parser.add_argument('--warmup', type=int, default=0.1,
                        help='warmup_ratio (default: 0.1)')
    parser.add_argument('--eval_steps', type=int, default=125,
                        help='eval_steps (default: 125)')
    parser.add_argument('--save_steps', type=int, default=125,
                        help='save_steps (default: 125)')
    parser.add_argument('--logging_steps', type=int,
                        default=25, help='logging_steps (default: 25)')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='weight_decay (default: 0.01)')
    parser.add_argument('--metric_for_best_model', type=str, default='accuracy',
                        help='metric_for_best_model (default: accuracy')
    
    args= parser.parse_args()

    return args

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    acc = accuracy_score(labels, preds) 

    return {
        'accuracy': acc,
    }

def train(args):
    
    
    seed_everything(args.seed)
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer= AutoTokenizer.from_pretrained(args.model)
    preprocess= Preprocess(args.train_path)

    all_dataset= preprocess.data
    all_label= all_dataset["grade"].values
    
    kfold= StratifiedKFold(n_splits= args.fold, shuffle= True, random_state= 42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_dataset, all_label)):
        run= wandb.init(project= 'essay', entity= 'quarter100', name= args.wandb_path, group = 'krl_grade_classifier')
        print(f'fold: {fold} start!')
        train_dataset= all_dataset.iloc[train_idx]
        val_dataset= all_dataset.iloc[val_idx]
        
        train_label= preprocess.label_to_num(train_dataset['grade'].values)
        val_label= preprocess.label_to_num(val_dataset['grade'].values)
        
        tokenized_train = preprocess.tokenized_dataset(train_dataset, tokenizer)
        tokenized_val= preprocess.tokenized_dataset(val_dataset, tokenizer)
        
        trainset= Dataset(tokenized_train, train_label)
        valset= Dataset(tokenized_val, val_label)
        
        model_config =  AutoConfig.from_pretrained(args.model)
        model_config.num_labels = 4

        model = AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config)
        model.to(device)
        
        save_dir= f'./result/{args.wandb_path}'

        training_args= TrainingArguments(
            output_dir= save_dir,
            save_total_limit= 1,
            gradient_accumulation_steps= args.gradient_accum,
            save_steps=args.save_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch_valid,
            label_smoothing_factor=0.1,
            warmup_ratio= args.warmup,
            weight_decay=args.weight_decay,
            logging_dir='./logs',
            logging_steps=args.logging_steps,
            metric_for_best_model= args.metric_for_best_model,
            evaluation_strategy= 'steps',
            group_by_length= True,
            eval_steps= args.eval_steps,
            load_best_model_at_end=True
        )

        if args.loss== 'LB':
            trainer= Trainer(
                model= model,
                args= training_args,
                train_dataset= trainset,
                eval_dataset= valset,
                compute_metrics= compute_metrics,
                callbacks= [EarlyStoppingCallback(early_stopping_patience= 3)]
            )

        elif args.loss== 'CE':
            trainer= Custom_Trainer(
                model=model,
                args=training_args,
                train_dataset=trainset,
                eval_dataset=valset,
                compute_metrics=compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience= 3)],
                loss_name = 'CrossEntropyLoss'
            )

        trainer.train()
        if not os.path.exists(f'{args.save_dir}'):
            os.makedirs(f'{args.save_dir}')
        torch.save(model.state_dict(), os.path.join(f'{args.save_dir}', 'pytorch_model.bin'))
        run.finish()
        print(f'fold{fold} fin!')
        break #1개의 fold만 사용


if __name__ == '__main__':

    args= get_config()
    train(args)
