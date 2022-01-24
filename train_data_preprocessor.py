import pandas as pd

def get_sentence_number(s):
    elements= s.split('-')
    print(elements)
    return int(elements[-1])

def get_text_name(s):
    elements= s.split('-')
    return '-'.join(elements[:-1])

file = pd.read_csv('data/training_data/train.csv')
file = file[['language','sentence_id','word_id','word','FFDAvg','FFDStd','TRTAvg','TRTStd','language_sentence_id','text_name']]
dev_file = pd.read_csv('data/training_data/dev.csv')
dev_file = dev_file[['language','sentence_id','word_id','word','FFDAvg','FFDStd','TRTAvg','TRTStd','language_sentence_id','text_name']]
df = pd.concat([file,dev_file])

df['text_name'] = df['language_sentence_id'].apply(get_text_name)
file['text_name'] = file['language_sentence_id'].apply(get_text_name)
dev_file['text_name'] = dev_file['language_sentence_id'].apply(get_text_name)
df.to_csv('data/training_data/train_and_valid.csv',index = False)
file.to_csv('data/training_data/train.csv',index = False)
dev_file.to_csv('data/training_data/dev.csv',index = False)
print(df['text_name'].unique())
