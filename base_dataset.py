from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class CreateDataset(Dataset):
    def __init__(self,data,labels,model):
        super().__init__()
        self.data = data
        self.labels = labels
        tokenizer = AutoTokenizer.from_pretrained(model)
        self.encodings  = tokenizer(data, add_special_tokens = True,  truncation = True, padding = "max_length", return_tensors = "pt",max_length=128)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        input_encoding = {key: val[index].clone().detach() for key, val in self.encodings.items()}
        return (input_encoding,torch.tensor(self.labels[index],dtype = torch.float))