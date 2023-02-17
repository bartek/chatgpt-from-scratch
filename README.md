Scripts and output from Andrej's fantastic lecture: https://www.youtube.com/watch?v=kCc8FmEb1nY

* simple_model.py is the first ~37 minutes of the video where we define and
  train a model which is very simple. The tokens are not talking to each other
  and are only looking at the very last character to make predictions of what
  comes next.

* single_attention.py is a model with single head of self-attention. This is up
  to around 1:20.

* multi_attention.py is a model with multi-headed self-attention and a bunch of
  other end of video optimizations. It is scaled up and takes a long time to
  run (~30 minutes on a M1 Macbook) and produces nonsensical but pretty good output!

* math_tricks.py is the various approaches to self attention and other things I
  barely understand! But it was fun to jot down notes for future thinking.
