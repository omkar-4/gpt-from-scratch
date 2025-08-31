import tiktoken
import torch

with open('training_data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("\nlength of dataset chars:", len(text))
# print("\ndataset preview:\n",text[:200]) # prints chars upto index 99

encoding = tiktoken.get_encoding('cl100k_base')

# -- example test for encoding/decoding --
# encoded_form = encoding.encode("hello bro, how are you? Alright?")
# print(encoded_form, len(encoded_form))

# decoded_form = encoding.decode(encoded_form)
# print(decoded_form, len(decoded_form))
# ----

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("\nvocab size:", vocab_size, "\nvocab chars:" ,''.join(chars))

stoi, itos = {}, {}
for i,ch in enumerate(chars):
    # print(i, ch)
    stoi[ch] = i
    itos[i] = ch
print("\nstoi, itos :", stoi, "\n\n",itos, "\n")

encode = lambda s: [stoi[c] for c in s] # str inp, int list out ; s:string
decode = lambda l: ''.join(itos[i] for i in l) # int list inp, str out

# test_str = "am gonna now goto school!"
# print("text:", test_str, "encoded text:", encode(test_str), "decoded text:", decode(encode(test_str)))

# print("\n\n test_str tokens:",encoding.encode(test_str), "\n\n test_str tokens decoded:",encoding.decode(encoding.encode(test_str)))

data = torch.tensor(encode(text), dtype=torch.long)
data_tto = torch.tensor(encoding.encode(text), dtype=torch.long)

print( "\n\n", data[:100], "\n\n" ,data_tto[:100])
print("\n\n data shape:", data.shape, "\n\n data_tto shape:", data_tto.shape, "\n\n", data.dtype, data_tto.dtype)

n = int(0.9 * len(data))
n_tto = int(0.9 * len(data_tto))

train_data = data[:n]
val_data = data[n:]

train_data_tto = data_tto[:n_tto]
val_data_tto = data_tto[n_tto:]

block_size = 8
train_data[:block_size+1]
train_data_tto[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
x_tto = train_data_tto[:block_size]
y_tto = train_data_tto[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

    context_tto = x_tto[:t+1]
    target_tto = y_tto[t]
    print(f"tiktoken : when input is {context_tto} the target: {target_tto}\n")

print("----------------------")

# --------------------------------

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will process in parallel
block_size = 8 # max context length for predictions

def get_batch(split):
    # general small batch of data if inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    print("ix:",ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    print(f"x:\n{x}\ny:\n{y}\n")
    return x,y

xb, yb = get_batch('train')
print('inputs:\n',"xb.shape:", xb.shape,"\nxb:\n", xb,"\n\ntargets:\n","yb.shape", yb.shape,"\nyb:", yb, "------\n\n")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target is: {target}")

print("\nlen of training data:",len(train_data), "\n")