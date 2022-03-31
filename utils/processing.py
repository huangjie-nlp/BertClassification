
import json
import pandas as pd

def generate_schema(fn):
    df = pd.read_csv(fn)
    sentence = df.sentence.tolist()
    label = df.label.tolist()
    label2id,id2label = {},{}
    for l in label:
        if l not in label2id:
            label2id[l] = len(label2id)
            id2label[len(id2label)] = l
    return label2id,id2label

if __name__ == '__main__':
    file = "../dataset/train_data.csv"
    label2id, id2label = generate_schema(file)
    json.dump([label2id,id2label],open("../dataset/schema.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)