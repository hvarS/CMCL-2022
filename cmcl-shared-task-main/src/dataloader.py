import torch
import transformers

FEATURES_NAMES = ['FFDAvg', 'FFDStd', 'TRTAvg', 'TRTStd']

class EyeTrackingCSV(torch.utils.data.Dataset):
  """Tokenize sentences and load them into tensors. Assume dataframe has sentence_id."""

  def __init__(self, df, mode = 'train',model_name='roberta-base'):
    self.model_name = model_name
    self.df = df.copy()
    self.mode = mode
    # Re-number the sentence ids, assuming they are [N, N+1, ...] for some N
    self.sentence_ids = self.df.sentence_id.unique()
    self.sentence_id_mapper = {}
    for i in range(len(self.sentence_ids)):
      self.sentence_id_mapper[i] = self.sentence_ids[i]
    self.num_sentences = len(self.sentence_ids)
    self.texts = []
    for id in self.sentence_ids:
      rows = self.df[self.df.sentence_id == id]
      text = rows.word.tolist()
      text[-1] = text[-1].replace('<EOS>', '')
      self.texts.append(text)
    # Tokenize all sentences
    if 'roberta' in model_name:
      self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    elif 'bert' in model_name:
      self.tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    self.ids = self.tokenizer(self.texts, padding=True, is_split_into_words=True, return_offsets_mapping=True)

  def __len__(self):
    return self.num_sentences
  

  def __getitem__(self, ix):
    input_ids = self.ids['input_ids'][ix]
    offset_mapping = self.ids['offset_mapping'][ix]
    attention_mask = self.ids['attention_mask'][ix]
    input_tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids]
    # First subword of each token starts with special character
    if 'xlm-roberta' in self.model_name:
      is_first_subword = [t[0] == '▁' for t in input_tokens]
    elif 'roberta' in self.model_name:
      is_first_subword = [t[0] == 'Ġ' for t in input_tokens]
    elif 'bert' in self.model_name:
      is_first_subword = [t0 == 0 and t1 > 0 for t0, t1 in offset_mapping]
    
    
    ls = []
    for i,val in enumerate(is_first_subword):
      if val:
        ls.append(i)

    if self.mode =='train' or self.mode =='val':
      features = -torch.ones((len(input_ids), 4))

      try:
        features[is_first_subword] = torch.Tensor(
          self.df[self.df.sentence_id == self.sentence_id_mapper[ix]][FEATURES_NAMES].to_numpy()
        )
      except:
        print('dataloader_train/val',ix,self.sentence_id_mapper[ix],len(ls))
        # for x,y in zip(input_tokens,is_first_subword):
        #   print(x,y)
        # raise Exception('Dataloader Length Not Matching')

      return (
        input_tokens,
        torch.LongTensor(input_ids),
        torch.LongTensor(attention_mask),
        features,
      )
    else:
      length = is_first_subword.count(True)
      features = -torch.ones((len(input_ids), 4))

      try:
        features[is_first_subword] = torch.zeros(4)
      except:
        print('data_loader_test',ix,self.sentence_id_mapper[ix],len(ls))

      return (
        input_tokens,
        torch.LongTensor(input_ids),
        torch.LongTensor(attention_mask),
        features,
      )
