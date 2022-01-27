import os

import pandas as pd

def joinToCreate(text,sentence):
    return '-'.join([text,sentence])

texts = ['ZuCo1' ,'ZuCo2', 'Provo', 'BSC' ,'RSC', 'PAHEC', 'PoTeC', 'GECO-NL']
files = {}

PRED_DIR = 'task1_predictions/'
for text in texts:
    file = pd.read_csv(PRED_DIR+text+'_predictions.csv')
    for i,row in file.iterrows():
        row['sentence_id'] = joinToCreate(row['text_name'],row['sentence_id'])
    print(file.head())
    break