from h11 import Data
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer


class TransduciveDataset(Dataset):
    def __init__(self,texts,labels,tf_name = 'bert-base-uncased') -> None:
        super(TransduciveDataset,self).__init__()
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tf_name)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        encoded_inputs = self.tokenizer(self.texts[index],padding = 'max_length',max_length = 32,truncation = True,return_tensors='pt',return_offsets_mapping=True)
        labels = self.labels[index] #[4,]
        decoded_texts = self.tokenizer.convert_ids_to_tokens(encoded_inputs.input_ids[0])
        word_mask = [t!='[CLS]' and t!='[SEP]' and t!='[PAD]' and t[0]!='#' for t in decoded_texts]
        return encoded_inputs,word_mask,labels

