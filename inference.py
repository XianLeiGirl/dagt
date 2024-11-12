from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader,DistributedSampler
import pandas as pd
from torch import optim
from torch import nn
import torch
from torch import amp
from torch.cuda.amp import GradScaler,autocast
import torch.distributed as dist
import torch.multiprocessing as mp
import os

import gc
gc.collect()
torch.cuda.empty_cache()


model_path = 'microsoft/deberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device('cuda:0')

max_len = 512
batch_size = 1
num_epoch = 1
learning_rage = 0.00001
world_size = 2

class MyDataset2(Dataset):
    def __init__(self, text_list, tokenizer, max_len):
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        text = self.text_list[index]
        tokenized = self.tokenizer(text=text,padding='max_length',max_length=self.max_len,truncation=True,return_tensors='pt')
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze()


predict_model = AutoModel.from_pretrained(model_path)

# 加载模型权重并移除 module 前缀
state_dict = torch.load('final_epoch.pth')
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
predict_model.load_state_dict(new_state_dict)

# predict_model.load_state_dict(torch.load('final_epoch.pth'))

predict_model.to(device)

test_df = pd.read_csv('./test_essays.csv')
test_texts = test_df['text']
test_dataset = MyDataset2(test_texts, tokenizer, max_len)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False )

predict_model.eval()
pred_prob = []
for (batch_input_ids, batch_attention_mask) in test_loader:
    batch_input_ids = batch_input_ids.to(device)
    batch_attention_mask = batch_attention_mask.to(device)
    output = predict_model(batch_input_ids, batch_attention_mask)
    # print("output:",output)

    pooled_output = output.last_hidden_state[:, 0, :]  # 形状为 (batch_size, hidden_size)
    # 进一步将隐藏层的特征缩减为一个值以进行二分类
    logits = nn.Linear(pooled_output.size(1), 1).to(device)  # 添加一个线性层
    # 将 pooled_output 输入线性层并调整 logits 的形状
    predictions = logits(pooled_output).squeeze() 
    print("predictions:",predictions)

    pred_prob.append(predictions.sigmoid().to('cpu').data.numpy().squeeze())

print("pred_prob:",pred_prob)    

id_list = test_df['id']
submission = pd.DataFrame(data={'id': id_list, 'generated': pred_prob})
submission.to_csv('submission.csv', index=False)
submission.head()    
