import math
from dataclasses import dataclass
from model import GPT
from torch.nn import functional as F
import torch
from model import GPTConfig
from pathlib import Path
from util import get_weights_file_path
torch.cuda.empty_cache()



# model = GPT.from_pretrained('gpt2') # loading weights from pretrained

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# torch.manual_seed(42)
torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)
elif device == "mps":
    torch.mps.manual_seed(1337)

import tiktoken


class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B*T
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


#optimization - works only for CUDA
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig()) # creating instance of untrained model
model.to(device)
if device == "cuda":
    model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr / 10
warmup_steps = 10
max_steps = 5000

# can use other LR scheduler like OCP
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


train_loader = DataLoaderLite(B = 12, T = 1024)

# expected initial loss = -ln(1/50257) ~ 10.9 ; 50257 is the total number of tokens, vocab size
import time
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
Path("weights").mkdir(parents=True, exist_ok=True)
scaler = torch.cuda.amp.GradScaler()
for step in range(max_steps):
    torch.cuda.empty_cache()
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    if device == "cuda":
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # does not work for mps
            logits, loss = model(x, y)
    else:
        logits, loss = model(x, y)
    # loss.backward()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # optimizer.step()
    scaler.step(optimizer)  # optimizer.step
    scaler.update()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f'step{step} | loss: {loss.item()} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec: .2f} | norm: {norm: .2f} | lr {lr}')
    if step > 0 and (step % 1000 == 0 or step == max_steps - 1):
        model_filename = get_weights_file_path(f"{step:02d}")
        torch.save(
            {
                "epoch": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
            model_filename
        )

print(loss)
