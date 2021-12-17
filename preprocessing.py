import pandas as pd
import re
import csv

def del_html(text):
    # 엔터 삭제
    text = re.sub("\n", '', text)
    
    # 괄호 삭제
    regex = r'\[[^)]*\]'
    text = re.sub(regex, '', text)
    
    regex = '<.*?>'
    text = re.sub(regex, ' ', text)
    
    # 기타 전처리
    text = re.sub("\xa0",'',text)
    text = re.sub(r'\ *\ ',' ',text) # 공백 여러개 하나로
    
    return text

data = pd.read_csv('./clear/writing/writing_paragraph.csv')
new_data = data['paragraph_txt']
str_list = []
header="paragraph_txt"
for i in new_data:
    new_str = del_html(i)
    str_list.append(new_str)

output_df = pd.DataFrame({'paragraph_txt' : str_list})
output_df.to_csv('new_writing_para.csv', index=False)
#new_data = del_html(data['paragraph_txt'])
#print(new_data)
