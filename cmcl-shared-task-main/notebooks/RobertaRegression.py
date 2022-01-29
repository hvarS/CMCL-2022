# %% [markdown]
# # RoBERTa Regression

# %%
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import torch
from collections import defaultdict, Counter
import random
import math
import pickle
import os

import src.eval_metric
import src.model
import src.dataloader

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

# %%
train_df = pd.read_csv("../../data/training_data/train.csv")
valid_df = pd.read_csv("../../data/training_data/dev.csv")

# %% [markdown]
# ## Fine-tune model

# %%
model_trainer = src.model.ModelTrainer(text_name='ZuCo2')

# %%
model_trainer.train(train_df, valid_df, num_epochs=150)

# %% [markdown]
# ## Make predictions

# %%
predict_df = model_trainer.predict(valid_df)
predict_df

# %%
predict_df.to_csv("predictions.csv", index=False)

# %%
src.eval_metric.evaluate(predict_df, valid_df)

# %%



