import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle 
import torch.optim as optim

## 模型定义
#(batch_size, num_heads, seq_len, d_k)
def scaled_dot_product_attention(query, key, value, attention_mask=None):
#     print(f'query.shape: {query.shape}, attention_mask.shape: {attention_mask.shape}')
    ## todo: q,k,v初始化
    d_k = query.size(-1)
    # scores:[batch_size, seq_len, seq_len]？
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if attention_mask is not None:
        scores = scores.masked_fill(attention_mask==0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)


    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        # [batch_size, sequence_length, d_model] -> [batch_size, self.num_heads, sequence_length, self.d_k]
        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.linear_q(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.linear_q(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
#             print(f'MultiHeadAttention.forward attention_mask unsqueeze: {attention_mask.shape}')
        attn_output, _ = scaled_dot_product_attention(q,k,v, attention_mask)

        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads*self.d_k)

        return self.linear_out(attn_output)


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # [seq_len, batch_size, d_model] + [seq_len, 1, d_model]
        return x+self.pe[:x.size(0), :]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
#         print(f'EncoderLayer.forward x.shape:{x.shape}, attention_mask.shape:{attention_mask.shape}')
        attn_output = self.attn(x,x,x,attention_mask)
        x = self.norm1(x+attn_output)
        ffn_output = self.ffn(x)
        return self.norm2(x+ffn_output)

class Detect(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, src_vocab_size, num_layers, max_len, num_labels):
        super(Detect, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionEncoding(d_model, max_len)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
#         self.encoder = Encoder(d_model, num_heads, d_ff, src_vocab_size, num_layers, max_len)
        self.fc = nn.Linear(d_model,num_labels)
        
    def forward(self, src, attention_mask=None):
#         print(f'Detect.forward src.shape:{src.shape}, attention_mask.shape:{attention_mask.shape}')
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for encoder in self.encoders:
            x = encoder(x, attention_mask)
#         x = self.encoder(src)

#         print(f'Detect x: {x}')
        output = self.fc(x)
#         print(f'Detect output: {output.shape}')
        return output


# prompt和text 涉及到语言
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 分词 input_ids+attention_mask
class CustomTokenize():
    def __init__(self):
        self.vocab = {}
         # start end符号
            
        self.vocab['[PAD]'] = 0
        self.vocab['[UNK]'] = 1
        
    # text -> vocab  
    def build_vocab(self, text):
        words = text.split()
        for word in words:
            if word not in self.vocab:
                self.vocab[word]=len(self.vocab)
        
    
    # text -> words -> input_ids+attention_mask
    def __call__(self, text, max_len):
        words = text.split()
        input_ids = [self.vocab.get(word, self.vocab['[UNK]']) for word in words] 
        
        attention_mask = [1] * len(input_ids)
        if len(input_ids) < max_len:
            padding_length = max_len-len(input_ids)
            input_ids = input_ids + [self.vocab['[PAD]']] * padding_length
            attention_mask = attention_mask + [0] * padding_length   
        else:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            
        ## unsqueeze    
#         print(f'CustomTokenize.call input_ids.shape:{torch.tensor(input_ids).shape}')
        return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}
   

# input_ids,attention_mask,label
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, labels=None):
        if isinstance(texts, pd.Series):
            self.texts = texts.tolist()
        else:
            self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokenize = self.tokenizer(self.texts[idx], self.max_len)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx])
        else:
            label=torch.tensor(-1)
        return {'input_ids': tokenize['input_ids'], 'attention_mask': tokenize['attention_mask'], 'label': label}


max_len = 256
batch_size = 4

def combine_text(df):
    df['combined_text'] = df[['instructions','source_text','text']].apply(lambda x: ''.join(x.dropna()), axis=1)
    return df

train_df = pd.read_csv('./train_essays.csv')
prompt_df = pd.read_csv('./train_prompts.csv')
merge_train = pd.merge(train_df, prompt_df, on='prompt_id')
merge_train = combine_text(merge_train)
# print(merge_train.head())

test_df = pd.read_csv('./test_essays.csv')
submission = pd.read_csv('./sample_submission.csv')
merge_test = pd.merge(test_df, prompt_df, on='prompt_id',how='left')
merge_test = combine_text(merge_test)
# print(merge_test.head())

tokenizer = CustomTokenize()
merge_train['combined_text'].apply(lambda x: tokenizer.build_vocab(x))
merge_test['combined_text'].apply(lambda x: tokenizer.build_vocab(x))

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)


with open('tokenizer.pkl', 'rb') as f:
    load_tokenizer = pickle.load(f)

   
## max_len
dataset = CustomDataset(merge_train['combined_text'], load_tokenizer, max_len, merge_train['generated'])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Detect(d_model=128, num_heads=8, d_ff=512, src_vocab_size=len(load_tokenizer.vocab), num_layers=6, max_len=max_len, num_labels=2)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) ## lr

print(f'len(dataloader): {len(dataloader)}')
for epoch in range(2):
    total_loss = 0
    for batch in dataloader:
        model.train()
#         print(batch['attention_mask'].shape)
        outputs = model(batch['input_ids'], batch['attention_mask'])
        logits = outputs[:,0,:]
#         print(f"logits.shape: {logits.shape}, outputs.shape: {outputs.shape}, batch['input_ids'].shape: {batch['input_ids'].shape}, batch['label'].shape: {batch['label'].shape}")
#         print(f'logits: {logits}')
        loss = criterion(logits, batch['label'])
        total_loss = total_loss + loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch+1}, loss: {total_loss / len(dataloader)}')
    
## 早停    
torch.save(model.state_dict(), 'final_model.pth')    

testset = CustomDataset(texts=merge_test['combined_text'], tokenizer=load_tokenizer, max_len=max_len)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

model = Detect(d_model=128, num_heads=8, d_ff=512, src_vocab_size=len(load_tokenizer.vocab), num_layers=6, max_len=max_len, num_labels=2)
model.load_state_dict(torch.load('final_model.pth'))
# print(model)
# for name, param in model.state_dict().items():
#     print(f'name: {name}, param: {param.shape}')

predicts = []
model.eval()
with torch.no_grad():
    for batch in testloader:
        ## todo: test_text过小
        output = model(batch['input_ids'], batch['attention_mask'])
        first_output = output[:,0,:]
#         print(f'output: {first_output}')
        predict,_ = torch.max(F.softmax(first_output, dim=-1),dim=1)
#         print(f'predict: {predict}')
        predicts.append(predict)

predicts = torch.cat(predicts)
predicts = torch.nan_to_num(predicts, nan=0)

predict_df = pd.DataFrame(predicts, columns=['generated'])
df = pd.concat([test_df, predict_df], axis=1)

submission = df[['id','generated']]
submission.to_csv('submission.csv')

submission.head()
