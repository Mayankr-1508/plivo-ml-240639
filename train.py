import math
import torch
from model import GPTLanguageModel
from tokenizer import ByteTokenizer

# Lightweight configuration strictly under 2M parameters
batch_size = 32
block_size = 256
max_iters = 2000
learning_rate = 3e-4
weight_decay = 0.1
eval_interval = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('train_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = ByteTokenizer()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

model = GPTLanguageModel(vocab_size=256, block_size=block_size, n_embd=192, n_head=4, n_layer=4, dropout=0.1)
model.to(device)

print(f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def get_lr(it):
    warmup_iters = 100
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return learning_rate * 0.1
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * (0.1 + 0.9 * coeff)

for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        print(f"Step {iter}: train loss {loss.item():.4f}, lr {lr:.5f}")

# Save checkpoint including config dictionary for evaluate.py
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': 256,
        'block_size': block_size,
        'n_embd': 192,
        'n_head': 4,
        'n_layer': 4,
        'dropout': 0.1
    }
}
torch.save(checkpoint, 'ckpt.pt')
print("Training completed and checkpoint saved with config dictionary as ckpt.pt")
