import pandas as pd
import re
import csv
from hanspell import spell_checker
from tqdm import tqdm
from sklearn.metrics import f1_score
from pykospacing import Spacing

data = pd.read_csv('./new_agree_para.csv')
new_data = data['paragraph_txt']
f1_list = []
cnt=1
for i in tqdm(new_data):
    print(cnt)
    tmp_list = i.split('#@문장구분#')
    spell_list = []
    space_list = []
    for t in tmp_list :
        try:
            spacing = Spacing()
            tmp = spacing(t)
            space_list.append(tmp)
            spelled_sent = spell_checker.check(tmp)
            hanspell_sent = spelled_sent.checked
            spell_list.append(hanspell_sent)
        except:
            a = t.replace("\t","")
            spacing = Spacing()
            tmp = spacing(a)
            space_list.append(tmp)
            spelled_sent = spell_checker.check(tmp)
            hanspell_sent = spelled_sent.checked
            spell_list.append(hanspell_sent)

    cor_list=[]
    pred_list=[]
    for cor, pred in zip(spell_list,space_list):
        c = cor.split(' ')
        p = pred.split(' ')
        for i,j in zip(c,p):
            cor_list.append(i)
            pred_list.append(j)
    
    f1 = f1_score(cor_list, pred_list, average='micro')
    f1_list.append(f1)
    cnt+=1

output_df = pd.DataFrame({'f1' : f1_list})
output_df.to_csv('spell_s_agree_para.csv', index=False)