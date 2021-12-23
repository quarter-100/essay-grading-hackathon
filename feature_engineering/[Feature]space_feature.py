import pandas as pd
import re
import csv
from hanspell import spell_checker
from tqdm import tqdm
from sklearn.metrics import f1_score
from pykospacing import Spacing

data = pd.read_csv('./new_df.csv')
new_data = data['paragraph_txt']

cnt=1
space_list = []
for i in tqdm(new_data):
    print(cnt)
    tmp_list = i.split('#@문장구분#')
    spaces = 0
    for t in tmp_list :
        try:
            spacing = Spacing()
            tmp = spacing(t)
            result = abs(len(tmp)-len(t))
        except:
            a = t.replace("\t","")
            spacing = Spacing()
            tmp = spacing(a)
            result = abs(len(tmp)-len(a))

        spaces += result

    space_list.append(spaces)
    cnt+=1

output_df = pd.DataFrame({'diff' : space_list})
output_df.to_csv('space_para.csv', index=False)