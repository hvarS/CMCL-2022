import pytorch_lightning as pl
from xlm_roberta import tfRegressor
import torch 
from dataloader import TransduciveDataLoader
import pandas as pd
import numpy as np

langTexts = ['ZuCo1','ZuCo2','Provo','BSC','RSC','PAHEC','PoTeC','GECO-NL']

tf_name = 'xlm-roberta-base'


train_loc = 'data/training_data/train.csv'
val_loc = 'data/training_data/dev.csv'
test_loc = 'data/test_data_subtask1/sub1/test_copy.csv'
pred_loc = 'data/test_data_subtask1/sub1/test.csv'
predictions_loc = 'data/task1_predictions/preds.csv'


dataloader = TransduciveDataLoader(train_loc,val_loc,test_loc,langTexts,tf_name)
trainer = pl.Trainer(gpus = [1],
                    max_epochs = 20)
model = tfRegressor(tf_name)

trainer.fit(model,datamodule=dataloader)
predictions = trainer.predict(model,datamodule=dataloader)


preds_file = pd.read_csv(pred_loc)
predictions = torch.cat(predictions)
print(len(predictions))
print(len(preds_file))
preds_file[['FFDAvg','FFDStd','TRTAvg','TRTStd']] = pd.DataFrame(np.array(predictions))

preds_file.to_csv(predictions_loc,index = False)



