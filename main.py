import torch
from base_model import Transformer
import pandas as pd
from base_dataset import CreateDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import mean

MAX_EPOCHS = 10

device = device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

transformer_model_pretrained = 'bert-base-multilingual-cased'

data = data = pd.read_csv('training_data2022/training_data/train_left_concat.csv')

sentences = list(data['left_sentences'])
labels = data[['FFDAvg', 'FFDStd','TRTAvg', 'TRTStd']].to_numpy()

x_train,x_val,y_train,y_val = train_test_split(sentences,labels,test_size=0.2)

train_dataset = CreateDataset(x_train,y_train,transformer_model_pretrained)
val_dataset = CreateDataset(x_val,y_val,transformer_model_pretrained)


train_dataloader = DataLoader(train_dataset,batch_size=8,shuffle = True)
val_dataloader = DataLoader(val_dataset,batch_size=8,shuffle = True)



model = Transformer(transformer_model_pretrained,num_classes = 4)

##Shift to CUDA backend
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr = 1e-3)
loss_fn = torch.nn.MSELoss()


for epoch in range(MAX_EPOCHS): 
    print(f"Epoch {epoch+1}\n-------------------------------")

    ############-------Train------##############
    size = len(train_dataloader.dataset)
    model.train()
    losses = []
    for idx,batch in enumerate(tqdm(train_dataloader)):
        x = batch[0]
        x = {key:value.to(device) for key,value in x.items()}
        y = batch[1].to(device)

        y_pred = model(x)
        loss = loss_fn(y_pred,y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    loss = mean(losses)
    print('-----------Training Loss---------------::',loss)

    ############-------Validation-----##############
    size = len(val_dataloader.dataset)
    losses = []
    model.eval()
    with torch.no_grad():
        for idx,batch in enumerate(tqdm(train_dataloader)):
            x = batch[0]
            x = {key:value.to(device) for key,value in x.items()}
            y = batch[1].to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred,y)

            losses.append(loss.item())
    loss = mean(losses)
    print('-----------Validation Loss---------------::',loss)


