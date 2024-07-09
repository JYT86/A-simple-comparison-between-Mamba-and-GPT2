from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
import torch
from data import train_loader, val_loader, vocab_size, tokenizer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
config = GPT2Config(vocab_size=vocab_size, n_positions=1024, n_embd=128, n_layer=3, n_head=8)
model = GPT2LMHeadModel(config).to(device)
# print(model)
total = len(train_loader)*100
bar = tqdm(total=total)
loss_func = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
for i in range(100):
    l = 0
    for x, y in train_loader:
        batch = x.to(device)
        logits = model(batch).logits.to(device)

        shifted_logits = logits
        shifted_labels = y.to(device)

        loss = loss_func(shifted_logits.reshape((-1, config.vocab_size)), shifted_labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.update(1)
        l += loss
    if i % 10 == 0:
        l = l/len(train_loader)
        print(f"{i}, loss:{l}")

model.save_pretrained("checkpoints/gpt-on-textbook")
model.eval()
times = 10
topk = 50
with open("comparisons.txt", "a") as f:

    for i, (x, y) in enumerate(val_loader):
        prompt_ids_list = x[:, :10].tolist()[0]
        pred, gen_it = 0, 0
        print(tokenizer.decode(prompt_ids_list))
        f.write(tokenizer.decode(prompt_ids_list))
        f.write("\n")
        while pred != 102 and gen_it<100:
            gen_it += 1
            prompt_ids = torch.tensor(prompt_ids_list).to(device)
            prompt_ids = prompt_ids.unsqueeze(0)
            pred = model(prompt_ids).logits.reshape((-1, vocab_size))
            # pred = torch.argmax(pred[-1, :], -1).item()
            if topk != 0:
                pred = pred[-1, :]
                pred = F.softmax(pred, dim=-1)
                (values, indices) = torch.topk(pred, k=topk)
                pred[pred < values[-1]] = 0.
                pred = pred / pred.sum(axis=0, keepdims=True)
                pred = torch.multinomial(pred, num_samples=1).item()
            else:
                pred = torch.argmax(pred[-1, :], -1).item()
                
            prompt_ids_list.append(pred)
        
        print(tokenizer.decode(x.squeeze(0)))
        f.write(tokenizer.decode(x.squeeze(0)))
        f.write("\n")
        print(tokenizer.decode(prompt_ids_list))
        f.write(tokenizer.decode(prompt_ids_list))
        f.write("\n")

        if i >10:
            break

