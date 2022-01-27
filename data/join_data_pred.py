import os

import pandas as pd

def joinToCreate(text,sentence):
    return '-'.join([text,sentence])

texts = ['ZuCo1' ,'ZuCo2', 'Provo', 'BSC' ,'RSC', 'PAHEC', 'PoTeC', 'GECO-NL']
files = {}

PRED_DIR = 'task1_predictions/'
for text in texts:
    file = pd.read_csv(PRED_DIR+text+'_predictions.csv')
    ser = []
    for i,row in file.iterrows():
        ser.append(joinToCreate(row['text_name'],str(row['sentence_id'])))
    file['sentence_id'] = ser
    print(file.head())
    break