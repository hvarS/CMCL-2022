from transformers import AutoModel,AutoTokenizer 

texts = [['Safe','to','say'],['Way','to','much']]
tf_name = 'xlm-robert-base'

t = AutoTokenizer.from_pretrained(tf_name)
e = t(texts[0],padding = 'max_length',is_split_into_words = True,max_length = 128,truncation = True,return_tensors='pt')
for key in e.keys():
    print(e[key].shape)