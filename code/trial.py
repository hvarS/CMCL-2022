from transformers import AutoModel,AutoTokenizer 

texts = [['Safe','to','say'],['Way','to','much']]
tf_name = 'xlm-roberta-base'

t = AutoTokenizer.from_pretrained(tf_name)
m = AutoModel.from_pretrained(tf_name)
e = t(texts[0],padding = 'max_length',is_split_into_words = True,max_length = 128,truncation = True,return_tensors='pt')
e.input_ids = e.input_ids.unsqueeze(0)
e.attention_mask = e.attention_mask.unsqueeze(0)
print(e.input_ids.shape)
o = m(**e)