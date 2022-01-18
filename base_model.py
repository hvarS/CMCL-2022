  
from torch import nn
from transformers import AutoConfig, AutoModel

class Transformer(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()

        self.name = model

        config = AutoConfig.from_pretrained(self.name)
        config.output_hidden_states = True
        
        self.transformer = AutoModel.from_config(config)

        self.nb_features = self.transformer.pooler.dense.out_features

        self.pooler = nn.Sequential(
            nn.Linear(self.nb_features, self.nb_features), 
            nn.LeakyReLU(0.1),
        )

        self.logit = nn.Linear(self.nb_features, num_classes)

        

    def forward(self, encodings):

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        output = self.transformer(input_ids,attention_mask = attention_mask)
        hidden_states = output['hidden_states']
        
        hidden_states = hidden_states[-1][:, 0] # Use the representation of the first token of the last layer

        ft = self.pooler(hidden_states)

        return self.logit(ft)