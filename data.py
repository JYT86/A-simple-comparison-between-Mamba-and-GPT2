import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

with open("sales_textbook.txt", 'r') as f:
    text = f.read()
text = text.replace('\n', ' ')
sentences = text.split(".")
processed_sentences = [sen + "." for sen in sentences if len(sen)>40]

tokenizer = AutoTokenizer.from_pretrained('tokenzier_local')
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenized_input = tokenizer(processed_sentences, return_tensors="pt", padding="max_length", max_length=100)["input_ids"]
print("There are {} sentences in this dataset. ".format(tokenized_input.shape[0]))

vocab_size = len(tokenizer.vocab)
train_num = int(len(tokenized_input)*0.8)
train_ids = tokenized_input[:train_num]
val_ids = tokenized_input[train_num:]
class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.seq_len = data.shape[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index][:-1], self.data[index][1:]
    

train_set = SeqDataset(train_ids)
val_set = SeqDataset(val_ids)

train_loader = DataLoader(train_set, batch_size=20, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size = 1, shuffle=False)

# for batch in train_loader:
#     print(batch.shape)