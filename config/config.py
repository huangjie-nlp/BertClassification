
class Config():
    def __init__(self):
        self.label_num = 15
        self.cuda = 0
        self.dropout = 0.5
        self.epoch = 10
        self.batch_size = 32
        self.learning_rate = 1e-5
        self.step = 500
        self.label_flag = "stock"
        self.bert_path = "./bert-base-chinese"
        self.bert_dim = 768
        self.train_fn = "./dataset/train_data.csv"
        self.dev_fn = "./dataset/dev_data.csv"
        self.test_fn = "./dataset/test_data.csv"
        self.schema_fn = "./dataset/schema.json"
        self.log = "./log/{}_log.log"
        self.save_model = "./checkpoint/bertclassfication.pt"
        self.dev_result = "./dev_result/dev.csv"
        self.test_result = "./test_result/dev.csv"
