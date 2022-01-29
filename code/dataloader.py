from sys import getallocatedblocks
import pytorch_lightning as pl
import pandas as pd
from dataset import TransduciveDataset
from utils import getLangText,seperateHyphenToSentence


class TransduciveDataLoader(pl.LightningDataModule):
    def __init__(self,train_location,val_location):
        super().__init__()
        self.train_location = train_location
        self.val_location = val_location

    def prepare_data(self) -> None:
        self.train_df = pd.read_csv(self.train_location)
        self.val_df = pd.read_csv(self.val_df)
        return super().prepare_data()

    def wordsToSentences(self,df):
        self.df = df.copy()
        self.df.langText = self.df.sentence_id.apply(getLangText)
        self.df.sentence_id = self.df.sentence_id.apply(seperateHyphenToSentence)
        self.df = self.df[]
        # Re-number the sentence ids, assuming they are [N, N+1, ...] for some N
        self.df.sentence_id = self.df.sentence_id - self.df.sentence_id.min()
        self.num_sentences = self.df.sentence_id.max() + 1
        assert self.num_sentences == self.df.sentence_id.nunique()

        self.texts = []
        for i in range(self.num_sentences):
            rows = self.df[self.df.sentence_id == i]
            text = rows.word.tolist()
            text[-1] = text[-1].replace('<EOS>', '')
            self.texts.append(text)
