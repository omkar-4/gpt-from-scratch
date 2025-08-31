import torch

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
print(x,y)

# for t in range(block_size):
#   context = x[:t+1]
#   target= y[t]
#   print(f"when input is {context} target is {target}")