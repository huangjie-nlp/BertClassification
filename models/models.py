
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class BertClassification(nn.Module):
    def __init__(self,config):
        super(BertClassification, self).__init__()
        self.config = config
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.bert_dim,self.config.label_num)

    def forward(self,data):
        input_ids = data["input_ids"].to(self.device)
        mask = data["mask"].to(self.device)
        encode = self.bert(input_ids,attention_mask=mask)[0]
        cls = encode[:,:1]
        cls = self.dropout(cls)
        batch_size = input_ids.size(0)

        feature = cls.view([batch_size,-1])
        output = self.fc(feature)

        return F.log_softmax(output)
