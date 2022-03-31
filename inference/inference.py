
from models.models import BertClassification
import torch
import json
from transformers import BertTokenizer


class Inference():
    def __init__(self,config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")
        self.model = BertClassification(self.config)
        self.model.load_state_dict(torch.load(self.config.save_model,map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        with open(self.config.schema_fn,"r",encoding="utf-8") as f:
            self.id2label = json.load(f)[1]

    def __data_process(self,sentence):
        encode = self.tokenizer.encode_plus(sentence)
        input_ids = encode["input_ids"]
        mask = encode["attention_mask"]

        input_ids = torch.LongTensor([input_ids])
        mask = torch.LongTensor([mask])

        return {"input_ids":input_ids,
                "mask":mask}

    def predict(self,sentence):
        data = self.__data_process(sentence)
        pred = self.model(data)
        pred = pred.argmax(dim=-1).cpu().item()
        predict = self.id2label[str(pred)]
        print("label:",predict)