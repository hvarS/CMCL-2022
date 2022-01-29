import pandas as pd

def get_sentence_number(s):
    elements= s.split('-')
    print(elements)
    return int(elements[-1])

def get_text_name(s):
    elements= s.split('-')
    return '-'.join(elements[:-1])

df = pd.read_csv('data/test_data_subtask2/sub2/test.csv')
df['langText'] = df['sentence_id'].apply(get_text_name)
df['sentence_id'] = df['sentence_id'].apply(get_sentence_number)
df.to_csv('data/test_data/test_task2.csv',index = False)
