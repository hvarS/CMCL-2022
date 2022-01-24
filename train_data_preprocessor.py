import pandas as pd

file = pd.read_csv('training_data2022/training_data/train.csv')
dev_file = pd.read_csv('training_data2022/training_data/dev.csv')

df = pd.concat([file,dev_file])
df.to_csv('training_data2022/training_data/train_and_valid.csv',index = False)