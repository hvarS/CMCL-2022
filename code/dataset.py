import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MAX_LEN = 128

class TransduciveDataset(Dataset):
    def __init__(self,texts,labels,tf_name = 'xlm-roberta-base') -> None:
        super(TransduciveDataset,self).__init__()
        try:
            assert len(texts) == len(labels)
        except AssertionError:
            print(len(texts),len(labels))
        self.texts = texts
        self.labels = labels
        self.tf_name = tf_name
        self.tokenizer = AutoTokenizer.from_pretrained(tf_name)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        encoded_inputs = self.tokenizer(self.texts[index],padding = 'max_length',is_split_into_words = True,max_length = MAX_LEN,truncation = True,return_tensors='pt')
        labels = torch.zeros(MAX_LEN,4)
        decoded_texts = self.tokenizer.convert_ids_to_tokens(encoded_inputs.input_ids[0])
        if not 'xlm' in self.tf_name:
            word_mask = [t!='[CLS]' and t!='[SEP]' and t!='[PAD]' and t[0]!='#' for t in decoded_texts]
        else:
            word_mask = [t[0]=='▁' for t in decoded_texts]
        try:
            labels[word_mask] = torch.tensor(self.labels[index],dtype = torch.float) #[4,]
        except RuntimeError:
            print([(x,y) for x,y in zip(decoded_texts,word_mask)])
            raise 'Improper Tokenization '
        return encoded_inputs,word_mask,labels

