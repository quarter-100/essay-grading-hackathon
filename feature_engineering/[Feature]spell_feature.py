import pandas as pd
import re
import csv
from hanspell import spell_checker
from tqdm import tqdm
from sklearn.metrics import f1_score
from PyRouge.pyrouge import Rouge

data = pd.read_csv('./new_df.csv')
new_data = data['paragraph_txt']
f1_list = []
cnt=1
for i in tqdm(new_data):
    print(cnt)
    tmp_list = i.split('#@문장구분#')
    spell_list = []
    space_list = []
    r=Rouge()
    for t in tmp_list :
        try:
            spelled_sent = spell_checker.check(t)
            hanspell_sent = spelled_sent.checked
            spell_list.append(hanspell_sent)
        except:
            a = t.replace("\t","")
            a = a.replace("&","")
            spelled_sent = spell_checker.check(a)
            hanspell_sent = spelled_sent.checked
            spell_list.append(hanspell_sent)

    [precision, recall, f_score] = r.rouge_l(spell_list,tmp_list)
    f1_list.append(f_score)
    cnt+=1

output_df = pd.DataFrame({'Rouge_l_f1' : f1_list})
output_df.to_csv('spell_rouge.csv', index=False)