import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


data = pd.read_csv('training_data2022/training_data/train.csv')
print(data.keys())

list_of_sentences = []
sentence = ""
prev_sentence_id = None
for index, row in data.iterrows():
    if row['sentence_id'] == prev_sentence_id:
        sentence += row['word']
        list_of_sentences.append(sentence)
    else:
        sentence = ""
        sentence += row['word']
        list_of_sentences.append(sentence)
    sentence+=" "
    prev_sentence_id = row['sentence_id']

data["left_sentences"] = list_of_sentences 

labels = data[['FFDAvg', 'FFDStd','TRTAvg', 'TRTStd']]
