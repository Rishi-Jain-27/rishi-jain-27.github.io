---
layout: post
title: "A first LSTM in PyTorch, classifying MNIST as a sequence"
author: "Rishi Jain"
date: 2026-05-16
categories: projects
project_url: https://github.com/Rishi-Jain-27/rnn-learning
project_label: Code
---

After two convolutional projects, [MNIST]({% post_url 2026-05-12-mnist-classifier-first-pytorch-project %}) and [FER2013]({% post_url 2026-05-14-fer2013-classifier-when-transfer-learning-loses %}), the first video of a recurrent-network tutorial drops you straight back onto MNIST and asks you to classify the digits with an LSTM, which sounds like a step backwards until you see the trick in the framing. You take the 28×28 image and call each row a timestep, and now you are holding a sequence of length 28 with 28 features at every step, which is something a recurrent net can actually chew on. The code is on [GitHub](https://github.com/Rishi-Jain-27/rnn-learning), one `main.py`, two training epochs, and no grand claims attached. The point the exercise is quietly making, even though it never says so, is that an RNN does not care in the slightest whether the sequence is a natural one, and that if you can put your input into some order at all then you can feed it through.

## Image as sequence

The shape contortion is the whole conceptual move here. The dataloader hands you `(100, 1, 28, 28)`, a batch of 100, one channel, 28 rows, 28 columns, and a single reshape turns that into `(100, 28, 28)`, a batch of 100 with 28 timesteps of 28 features each. The top row of pixels becomes `t=0` and the bottom row becomes `t=27`, so that by the time the LSTM has consumed the final row it has, in principle, seen the whole digit in order. It is a slightly silly thing to do to an image, and it is also exactly the pattern recurrent nets get used for outside language, a known-length sequence going in and a single classification coming out, the many-to-one case. The choice of MNIST is the teaching part of it, because the dataset is solved and the labels are trivial and the only thing the exercise is really testing is whether you can wire the recurrent loop together correctly.

## Why LSTM, and not a vanilla RNN

The model reaches for `nn.LSTM` rather than `nn.RNN`, and the tutorial does not really defend the choice, leaning instead on the implicit defence that every recurrent-net tutorial leans on. Vanilla RNNs train badly past a small number of timesteps, because the gradient either explodes or vanishes as it travels back through the unrolled chain, and while 28 steps is not long it is not trivially short either, so there is no real reason to pay the vanilla-RNN tax on a learning exercise. I worked through where that explosion and vanishing actually come from in a [companion post on backprop through time]({% post_url 2026-05-16-bptt-vs-backprop-math %}), which is the math underneath this whole choice. The first small surprise is the signature of `nn.LSTM`, whose hidden state is a tuple rather than a single tensor: you initialise `h_0` and `c_0`, both zeros and both of shape `(num_layers, batch, hidden_size)`. The hidden state is what the network is computing at the current step, and the cell state is what it carries forward across steps, two separate channels of memory with gates between them, and that distinction is exactly what the L and the M in LSTM are buying you.

## Many-to-one: only the last step speaks

The call `out, _ = self.lstm(x, (h_0, c_0))` returns 28 hidden states, one for each row, and for classification the line that matters is the very next one, `out = out[:, -1, :]`, which keeps the final timestep and drops the other 27. The state at `t=27` has by construction integrated every earlier row into it, so the intermediate states would only matter if you were predicting at every step, the way you would for character-by-character language modelling or frame-by-frame tagging. The `_` in that unpack is the `(h_n, c_n)` tuple, the final hidden and cell states across all the layers, and for a unidirectional LSTM with `batch_first=True` the vectors `out[:, -1, :]` and `h_n[-1]` are literally the same thing. The code reaches for the output tensor rather than the discarded hidden state, which is the convention even if it is very slightly wasteful, and it is worth knowing both paths exist.

## batch_first is worth the argument

The default for `nn.LSTM` is `batch_first=False`, the sequence-first layout `(seq_len, batch, feature)`, while every other PyTorch layer expects batch-first, which makes the default a historical wart and nothing more. Setting `batch_first=True` on the layer and reshaping the input to match with `images.reshape(-1, sequence_length, input_size)` is two lines of consistency you have to remember to keep aligned, and the unpleasant part is what happens when you forget one of them, because the model still trains, just on garbage axes, with no traceback to tell you anything is wrong. You find out from a loss curve that never moves. The other quiet choice is `num_layers=2`, which stacks two LSTM cells with the output of the first feeding the input of the second, and the `num_layers` dimension in `h_0` and `c_0` is what plumbs the initialisation through to both of them. Deeper recurrent stacks are not free, since they amplify the very vanishing-gradient problem the LSTM gates are already fighting, but two layers is cheap enough to count as insurance.

## What this file doesn't cover

Some honest scope-setting is in order, given this is one video's worth of code. It runs for two epochs, enough to confirm the loss decreases and nothing more, with no best-checkpoint logic and no test-time evaluation beyond a single accuracy print at the end. It is unidirectional, walking the rows top to bottom and stopping there, even though a digit is just as legible read upward and `bidirectional=True` is a one-flag change worth understanding next. It has no real sequence in it, because MNIST-as-sequence is a crutch and the actual point of recurrent nets is data whose ordering genuinely means something, text and audio and time series, none of which is in this file yet. There is no GRU comparison either, though the GRU is the smaller cousin of the LSTM with one gate fewer and often comparable results, and it is worth a head-to-head later. And there is no gradient clipping, the standard hygiene for recurrent training through `torch.nn.utils.clip_grad_norm_`, which the tutorial skips because at 28 timesteps and 2 epochs it simply never bites.

## Takeaway

Recurrent nets feel deliberately backwards after a month of convolutions. A CNN sees the whole image at once and lets its filters discover the structure for themselves, whereas an LSTM walks across one slice at a time with the gates deciding what to carry forward, and the 28-row reshape is what bridges those two views of the same pixels. It is a useful warm-up and not a real application, and the real application is the next video.
