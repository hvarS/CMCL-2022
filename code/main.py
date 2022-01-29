import pytorch_lightning as pl
from xlm_roberta import tfRegressor
from dataloader import TransduciveDataLoader

langTexts = ['ZuCo1','ZuCo2','Provo','BSC','RSC','PAHEC','PoTeC','GECO-NL']

tf_name = 'xlm-roberta-base'

train_loc = 'data/training_data/train.csv'
val_loc = 'data/training_data/dev.csv'
test_loc = ''
langText = langTexts[5]


dataloader = TransduciveDataLoader(train_loc,val_loc,langText,tf_name)
trainer = pl.Trainer(gpus = [1],
                    max_epochs = 15)
model = tfRegressor(tf_name)

trainer.fit(model,datamodule=dataloader)


