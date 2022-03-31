# BertClassfication
使用bert finetune做文本分类任务

## 运行
1、下载bert并放入bert-base-chinese文件夹下  
2、config/config.py配置模型参数  
3、log保存日志  
4、python main.py训练模型，并且完成自测结果  
5、python test.py 单句预测

## 实验环境
torch == 1.7.1  
transfomers == 3.4.0  
tqdm == 4.59.0  
pandas == 1.1.5