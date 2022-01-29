import pytorch_lightning as pl
from transformers import AutoModel,AutoTokenizer
import torch

class tfRegressor(pl.LightningModule):
    def __init__(self,tf_name):
        super(tfRegressor,self).__init__()
        self.fe = AutoModel.from_pretrained(tf_name)
        self.regressor = torch.nn.Linear(768,4)
        self.criterion = torch.nn.L1Loss()
        
    def forward(self,batch):
        encoded_inputs = batch[0] #{i/p_ids,attention_masks}
        word_masks = batch[1] #[b,128]
        labels = batch[2] #[b,128,4]
        labels_mask = batch[3] #[b,128]
        for key in encoded_inputs.keys():
            encoded_inputs[key] = encoded_inputs[key].squeeze()
        outputs = self.fe(**encoded_inputs)
        outputs = outputs.last_hidden_state #[b,128,768]
        preds = self.regressor(outputs) #[b,128,4]

        ##Masking to generate equal number y_pred and y_true
        y_pred = preds[word_masks]
        y_true = labels[labels_mask]

        assert y_pred.shape == y_true.shape
        return y_pred,y_true
    def training_step(self,batch,idx):
        y_pred,y_true = self(batch)
        loss = self.criterion(y_pred,y_true)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.02)
    


