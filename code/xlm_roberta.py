import pytorch_lightning as pl
from transformers import AutoModel,AutoTokenizer
import torch
import numpy as np

class tfRegressor(pl.LightningModule):
    def __init__(self,lr,tf_name):
        super(tfRegressor,self).__init__()
        self.fe = AutoModel.from_pretrained(tf_name)
        self.lr = lr
        self.regressor = torch.nn.Linear(768,4)
        self.criterion = torch.nn.L1Loss()
        
    def forward(self,encoded_inputs):
        outputs = self.fe(**encoded_inputs)
        outputs = outputs.last_hidden_state #[b,128,768]
        preds = self.regressor(outputs) #[b,128,4]
        return preds
        
    def training_step(self,batch,idx):
        encoded_inputs = batch[0] #{i/p_ids,attention_masks}
        word_masks = batch[1] #[b,128]
        labels = batch[2] #[b,128,4]
        for key in encoded_inputs.keys():
            encoded_inputs[key] = encoded_inputs[key].squeeze()
        preds = self(encoded_inputs)
        ##Masking to generate equal number y_pred and y_true
        preds[word_masks == 0] = -1
        y_pred = preds
        y_true = labels

        assert y_pred.shape == y_true.shape
        loss = self.criterion(y_pred,y_true)
        
        return loss
    
    def validation_step(self,batch,idx):
        encoded_inputs = batch[0] #{i/p_ids,attention_masks}
        word_masks = batch[1] #[b,128]
        labels = batch[2] #[b,128,4]
        for key in encoded_inputs.keys():
            encoded_inputs[key] = encoded_inputs[key].squeeze()
        preds = self(encoded_inputs)
        ##Masking to generate equal number y_pred and y_true
        preds[word_masks == 0] = -1
        y_pred = preds
        y_true = labels

        assert y_pred.shape == y_true.shape
        loss = self.criterion(y_pred,y_true)
        
        return loss  

    def validation_epoch_end(self, outputs):
        print('val loss ',torch.mean(torch.stack(outputs)))
    def predict_step(self, batch,batch_idx):
        encoded_inputs = batch[0] #{i/p_ids,attention_masks}
        word_masks = batch[1] #[b,128]
        labels = batch[2] #[b,128,4]
        for key in encoded_inputs.keys():
            encoded_inputs[key] = encoded_inputs[key].squeeze()
        preds = self(encoded_inputs)
        
        return preds[word_masks]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    


