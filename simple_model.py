import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# When we typically train a transformer, we work with chunks of the dataset.
# Train chunks at a time.
block_size = 8 # Sometimes referenced as context length
batch_size = 4 # number of chunks of text we want to process at a time

# As we are sampling these chunks of text, we are going to have many batches of
# chunks of text which are stacked into a single tensor. This is mostly done
# for efficiency. These are processed completely independently
torch.manual_seed(1337)

# Bigram is the simplest neural network
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # nn.Embedding is a very thin wrapper around shape (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # Batch by time by channel tensor (B, T, C)
        logits = self.token_embedding_table(idx) 

        # Reshape
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Now that we can make predictions about what comes next, we need to
            # evaluate the loss function. A good way is to use the negative log
            # likelihood loss.  This measures the quality of the logits. We have
            # the identity of the next character, so how well are we predicting
            # the next character based on the logits?
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Now that we can evaluate the quality of the model we also wish to
    # generate from the model. In this function we take the same input (idx).
    # The job of generate is to take a B, T and make it B,T + 1, +2, +3 .. as
    # many as want (max_new_tokens)
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# input.txt generated from https://huggingface.co/datasets/tiny_shakespeare
# https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('length of dataset in characters', len(text))
print(text[:1000])

chars = sorted(list(set(text))) # all the unique characters in the text
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# The vocabulary is the possible characters the model can emit
# Tokenize to some sequence of integers to a vocabulary of possible elements
# Translate individual characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join(itos[i] for i in l) # decoder: take a list of integers and output a string 

print(encode("hii there"))
print(decode(encode("hii there")))

# Let's now encode the entire text dataset and store it in a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# Now let's split up the data into train and validation sets
# We will withhold 10% of it (this is the validation data). This will help us
# understand to what extend our model is overfitting. We don't want a perfect
# memorization of Shakespeare -- rather, we want a neural network which creates
# shakespeare-like text.
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


print(train_data[:block_size+1]) # 9 characters of the training set

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-----')


# Let's start feeding this into the simplest neural network
m = BigramLanguageModel(vocab_size)
out, loss = m(xb, yb)

# We get the predictions (or scores or logitcs) for every 4x8 position
print(out.shape, loss)

# Let's generate now!
# 
# Batch and time will be 1,1. It is holding a zero. Zero is how we kick off the
# generation. Remember that 0 is standing in for a newline character. It is a
# reasonable character to kick off in a sequence.
#
# We ask for 100 tokens.
# Generate works in 0 batches, so we pluck the 0-index. That gives us a
# timestep, which is specific point in time within a sequence, where each
# timestep corresponds to e.g. a specific word in the sequence.
#
# Then can feed into the decode function and convert those integers into
# text
#
# It is garbage! Why? This is because the model is totally random. It has not
# been trained.
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# Now let's train the model. First we create a PyTorch optimization object.
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32

# This is a typical training loop, some number of steps (100)
for steps in range(10000):
    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss =  m(xb, yb)

    # Zero out gradients from previous step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Use those gradients to update the parametets
    optimizer.step()



# In 100 loops, we expect to start a bit higher and loss goes lower. From ~4.7 to ~4.3
# We want to train for longer. If the loss is improving we can expect something
# a bit more reasonable
print(loss.item())

print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# We can continue playing with this to get something reasonable-ish
# This is the simplest possible model. This is very simple becausse the tokens
# are not talking to each other. We are only looking at the very last character
# to make predictions of what comes next.
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
