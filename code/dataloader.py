import pytorch_lightning as pl
import torch.utils.data as TorchData
import pandas as pd
from dataset import TransduciveDataset
from utils import getLangText,seperateHyphenToSentence
import numpy as np


class TransduciveDataLoader(pl.LightningDataModule):
    def __init__(self,train_location,val_location,test_loc,langTexts,tf_name):
        super().__init__()
        self.train_location = train_location
        self.val_location = val_location
        self.test_location = test_loc
        self.langTexts = langTexts
        self.tf_name = tf_name

    def prepare_data(self) -> None:
        self.train_df = pd.read_csv(self.train_location)
        self.val_df = pd.read_csv(self.val_location)
        self.predict_df = pd.read_csv(self.test_location)
        self.train_dataset = self.datasetGen(self.train_df)
        self.val_dataset = self.datasetGen(self.val_df)
        self.predict_dataset = self.datasetGenPred(self.predict_df)
        return super().prepare_data()

    def datasetGen(self,df):
        self.df = df.copy()
        self.df['langText'] = self.df.sentence_id.apply(getLangText).astype(str)
        self.df.sentence_id = self.df.sentence_id.apply(seperateHyphenToSentence)
        self.df.sentence_id = self.df.sentence_id.astype(int)
        self.texts = []
        self.labels = []
        for langText in self.langTexts:
            df = self.df.copy()
            df = df[df.langText == langText]
            texts = []
            labels = []
            for i in df.sentence_id.unique():
                rows = df[df.sentence_id == i]
                label = rows[['FFDAvg','FFDStd','TRTAvg','TRTStd']].to_numpy()
                text = rows.word.tolist()
                texts.append(text)
                labels.append(label)
            self.texts.extend(texts)
            self.labels.extend(labels)
        self.labels = np.array(self.labels)
        return TransduciveDataset(self.texts,self.labels,self.tf_name)

    def datasetGenPred(self,df):
        self.df = df.copy()
        self.df['langText'] = self.df.sentence_id.apply(getLangText).astype(str)
        self.df.sentence_id = self.df.sentence_id.apply(seperateHyphenToSentence)
        self.df.sentence_id = self.df.sentence_id.astype(int)
        self.texts = []
        self.labels = []
        for langText in self.langTexts:
            df = self.df.copy()
            df = df[df.langText == langText]
            texts = []
            labels = []
            for i in df.sentence_id.unique():
                rows = df[df.sentence_id == i]
                label = np.zeros((len(rows),4))
                text = rows.word.tolist()
                texts.append(text)
                labels.append(label)
            self.texts.extend(texts)
            self.labels.extend(labels)
        self.labels = np.array(self.labels)
        return TransduciveDataset(self.texts,self.labels,tf_name = self.tf_name,mode = 'test')

    def train_dataloader(self):
        return TorchData.DataLoader(self.train_dataset,batch_size=16,shuffle = True)
    def val_dataloader(self):
        return TorchData.DataLoader(self.val_dataset,batch_size=16)
    def predict_dataloader(self):
        return TorchData.DataLoader(self.predict_dataset,batch_size=16)

    

# train_loc = 'data/training_data/train.csv'
# val_loc = 'data/training_data/dev.csv'
# langText = 'ZuCo1'
# dataloader = TransduciveDataLoader(train_loc,val_loc,langText)
# dataloader.prepare_data()

# for batch in dataloader.train_dataloader():
#     enc_inputs,word_mask,labels= batch[0],batch[1],batch[2]
#     break
