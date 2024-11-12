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

class MyDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_len, label_list):
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_list = label_list

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        text = self.text_list[index]
        label = self.label_list[index]
        tokenized = self.tokenizer(text=text,padding='max_length',max_length=self.max_len,truncation=True,return_tensors='pt')
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), label

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

# hyperparameters
max_len = 512
batch_size = 2
num_epoch = 1
learning_rage = 0.00001
world_size = 2

model_path = 'microsoft/deberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

train_df = pd.read_csv('./train_essays.csv')
train_texts = train_df['text']
train_labels = train_df['generated']
train_dataset = MyDataset(train_texts, tokenizer, max_len, train_labels)
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=2,pin_memory=False ) ## 并发


def train(rank, num_epoch, model, train_dataset, device, batch_size):
    # 设置设备
    device = torch.device(f'cuda:{rank}')

    # 初始化分布式进程组，torchrun会自动设置 RANK 和 WORLD_SIZE
    dist.init_process_group("nccl")

    # 将模型放到对应的 GPU 上
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 定义优化器、损失函数和混合精度缩放器
    optimizer = optim.Adam(model.parameters(), lr=learning_rage)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # 使用分布式采样器
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)

    # 开始训练
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
            # 将数据加载到设备
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.float().to(device)

            # 混合精度训练
            with autocast():
                output = model(batch_input_ids, batch_attention_mask)
                # print("output:", output)

                # 获取 batch_size 个样本的隐藏状态的第一个标记
                pooled_output = output.last_hidden_state[:, 0, :]  # 形状为 (batch_size, hidden_size)

                # 进一步将隐藏层的特征缩减为一个值以进行二分类
                logits = nn.Linear(pooled_output.size(1), 1).to(device)  # 添加一个线性层

                # 将 pooled_output 输入线性层并调整 logits 的形状
                predictions = logits(pooled_output).squeeze()  # 形状为 (batch_size)

                # 计算损失
                loss = loss_fn(predictions, batch_labels)

                # loss = loss_fn(output.last_hidden_state[:, 0, :].reshape(-1), batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Rank {rank}, Epoch [{epoch}/{num_epoch}], Loss: {total_loss / len(train_dataloader)}")

    # 销毁进程组
    dist.destroy_process_group()

    torch.save(model.state_dict(), 'final_epoch.pth')


def main():
    # 获取当前进程的 rank
    rank = int(os.environ['RANK'])
    train(rank, num_epoch, model, train_dataset, device, batch_size)

if __name__ == "__main__":
    main()



# torchrun --nproc_per_node=1 train.py
