import torch
import torch.nn as nn
from torch.nn import functional as F

with open('training_data/input.txt', 'r', encoding="utf-8") as f:
  text = f.read()

# print(f"\n length of dataset chars: {len(text)} \n")
# print(f"\n preview: \n\n {text[:100]} \n")

chars = sorted(list(set(text))) # returns [] of comma sep char str in asc order
vocab_size = len(chars)

# print(f"\n chars: {chars} \n\n vocab_size:{vocab_size} \n\n char condensed : {''.join(chars)} \n ")

# --- building an index-based encoder/decoder
# enum extracts index,value and creates dict with key:val pairs
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # takes str returns [int,]
decode = lambda l: ''.join([itos[i] for i in l]) # takes [int,] returns str

# testing encoder and decoder
# print(encode("how are you?"))
# print(decode(encode("how are you?")))

data = torch.tensor(encode(text), dtype=torch.long) # dtype is 64-bit int
# data is a rank 1 tensor (1D vector)

# print("data tensor:", data[:100])
# print("data shape:", data.shape, data.dtype)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1] # +1 as I need 1 more token for y's offset, offset is for prediction of next char

x = train_data[:block_size]
y = train_data[1:block_size+1]
# print(x,y)

# for t in range(block_size):
#   context = x[:t+1]
#   target= y[t]
#   print(f"when input is {context} target is {target}")

# the model must consider not just current letter but all letters before it as well, hence it uses entire prev context to predict next/target char

print("---------") # --------------------------------------------------

# --- training loop

torch.manual_seed(1337) # for setting RNN
block_size = 8 # max context length for predictions
batch_size = 4 # how many independent sequences will process in parallel

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) # pick batch_size no. of random pos from data (90% train or 10% val)
  # print("ix", ix)
  # print("data :",data)

  x = torch.stack([data[i:i+block_size] for i in ix]) # iterate over ix, from each pos extract block_size chars
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # iterate over ix, from each pos with 1 offset extract block_size+1 chars
  # print(f"x:{x} \n\n y:{y}")
  return x,y

xb, yb = get_batch(batch_size)
# print(f'\n xb:{xb} \n\n yb:{yb}, \nxb_shape: {xb.shape}, yb_shape: {yb.shape}')

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target is: {target}")

# print("\nlen of training data:",len(train_data), "\n")


# --- bigram model
# torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

  def __init__(self, vocal_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets):
    # idx and targets are both (B,T) tensor of integers
    logits = self.token_embedding_table(idx) # (B,T,C) - batch, time, channels (tensor)
    # here : batch = 4, time = 8, channels = vocal_size = 65
    B,T,C = logits.shape
    logits = logits.view(B*T, C)

    # targets = targets.view(-1)
    targets = targets.view(B*T)

    loss = F.cross_entropy(logits, targets) # logits are predictions

    # logits are scores or next chars in seq
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      # get predictions
      logits, loss = self(idx)
      # focus only on the last time step 
      logits = logits[:,-1,:] # becomes (B,C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B,C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
    return idx
  
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)

print("logits.shape :", logits.shape)
print("loss:", loss)
