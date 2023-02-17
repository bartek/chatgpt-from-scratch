# The mathematical trick in self-estimation
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# What is the easiest way for tokens to communicate?
# Just do an average of all the preceeding elements. If I am the 5th token, I
# can take the channels which make up the information that make up my step, as
# well as the channels from steps 4, 3, 2, 1 and take the average. This becomes
# a feature vector which summarizes me in the context of my history!


# We're going to create x, and bow. bow is short for bag of words
# Initialize it at 0, iterate over batches, and time
# The result is a little vector we can store in xbow
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b, t] = torch.mean(xprev, 0)

print(x[0])
print(xbow[0])

# We can be more efficient by using matrix multiplication
# In particular, we can produce an array wei (of weights). This is how much of
# every row we want to average up
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
print(wei)

# Think this through: wei is (T, T) @ (B, T, C) -- > (B, T, C)
xbow2 = wei @ x # (T, T)

# xbow and xbow2 should be identical
assert torch.allclose(xbow, xbow2)

# Another option. Use Softmax. Identical to the rest otherwise
# The reason we use this one is the weights begin with 0 (think of it as an
# interaction string, which is a set of variables that are used together to
# capture the interactions or relationships between them)
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
xbow3 = wei @ x
assert torch.allclose(xbow, xbow3)

# Crux: We're going to implement a small self-attention for a single head
# The problem self attention solves can be summarized as: I am a vowel then
# maybe I am looking for constanants in the past and I want to know what those
# consts are and flow it to me. I want to now gather information from the past
# and gather it in a data dependent way
#
# Every single token at each position will emit two vectors. It will emit a
# query and a key. The query vector roughly speaking is "What am I looking
# for?" and the key vector is "What do I contain?"
#
# And then the way we get affinities is by taking the dot product of the query
# and keys
torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch, time, channels
x = torch.randn(B, T, C)

# Let's implement a single head of self attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# Forward these modules onto x
k = key(x) # (B, T, 16) because 16 is head size 
q = query(x) # (B, T, 16)

# We want the affinites of these to be:
# For every row of B we will have a T x T matrix
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))

# in encoder blocks, you'd delete this to allow all the nodes to talk. in
# decoder always present (to get the triangular structure) 
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

# v is the vector we aggregate instead of raw x
v = value(x) 
out = wei @ v
print(out)

# Notes about self attention:
# Attention is a communication mechanism. Can be seen as nodes in a directed
# graph looking at each other and aggregating information with a weighted sum
# from all nodes that point to them, with data-dependent weights.
#
# There is no notion of space. Attention simply acts over a set of vectors.
# This is why we need to positionally encode tokens.
#
# Each example across batch dimension is of course processed completely
# independently and never "talk" to each other
#
# In an "encoder" attention block just delete the single line that does masking
# with tril, allowing all tokens to communicate. This block here is called a
# "decoder" attention block because it has triangular masking, and is usually
# used in autoregressive settings, like language modeling.
#
# "self-attention" just means that the keys and values are produced from the
# same source as queries. In "cross-attention", the queries still get produced
# from x, but the keys and values come from some other, external source (e.g.
# an encoder module)
#
# "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it
# so when input Q,K are unit variance, wei will be unit variance too and
# Softmax will stay diffuse and not saturate too much. Illustration below

# Cross Attention?
#
# Self attention means keys, queries, and values come from the same source (x)
#
# In principle, attention is much more general than that. You can have queries
# produced from x but the keys and valeus come from a different external source
#
# Produce queries and read from the side. Cross attention when there is a
# separate source of nodes we want to pool information from. 
