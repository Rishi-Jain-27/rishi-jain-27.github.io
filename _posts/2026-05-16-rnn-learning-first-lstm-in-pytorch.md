---
layout: post
title: "A first LSTM in PyTorch, classifying MNIST as a sequence"
author: "Rishi Jain"
date: 2026-05-16
categories: projects
project_url: https://github.com/Rishi-Jain-27/rnn-learning
project_label: Code
---

After two convolutional projects ([MNIST]({% post_url 2026-05-12-mnist-classifier-first-pytorch-project %}) and [FER2013]({% post_url 2026-05-14-fer2013-classifier-when-transfer-learning-loses %})), the first video of a recurrent-network tutorial drops you back on MNIST and tells you to classify the digits with an LSTM. The trick is the framing: take the 28×28 image and call each row a timestep. Now you have a sequence of length 28 with 28 features per step, and a recurrent net has something to chew on. The code is on [GitHub](https://github.com/Rishi-Jain-27/rnn-learning): one `main.py`, two training epochs, no claims.

The thesis the exercise is making, even if it doesn't say so, is that an RNN doesn't care whether the sequence is natural. If you can order your input, you can feed it through.

## Image as sequence

The shape contortion is the whole conceptual move. The dataloader hands you `(100, 1, 28, 28)`: batch of 100, one channel, 28 rows, 28 columns. One reshape and it becomes `(100, 28, 28)`: batch of 100, 28 timesteps, 28 features each. Top row of pixels is `t=0`, bottom row is `t=27`. By the time the LSTM has consumed the final row it has, in principle, seen the whole digit in order.

This is silly and it's also exactly the pattern recurrent nets get used for outside NLP: known-length sequence in, single classification out. Many-to-one. The choice of MNIST is the teaching part. The dataset is solved, the labels are trivial, and the only thing the exercise is testing is whether you can wire the recurrent loop together.

## Why LSTM, not vanilla RNN

The model uses `nn.LSTM`, not `nn.RNN`, and the tutorial doesn't really defend the choice. The implicit defense is the same one every recurrent-net tutorial relies on. Vanilla RNNs train badly past a small number of timesteps because the gradient either explodes or vanishes through the unrolled chain. 28 isn't long, but it isn't trivially short either, and there's no reason to pay the vanilla-RNN tax on a learning exercise.

The signature is the small surprise. `nn.LSTM`'s hidden state is a tuple, not a tensor. You initialize `h_0` and `c_0`, both zeros, both of shape `(num_layers, batch, hidden_size)`. Hidden state is what the network is computing at this step. Cell state is what it is carrying forward across steps. Two separate channels of memory, gated. That distinction is what the L and the M in LSTM are buying you.

## Many-to-one: only the last step speaks

`out, _ = self.lstm(x, (h_0, c_0))` returns 28 hidden states, one per row. For classification, the line that matters is the next one: `out = out[:, -1, :]`. Keep the final timestep, drop the other 27. The state at `t=27` has, by construction, integrated every earlier row into it. The intermediate states would only matter if you were predicting at every step (character-by-character language modelling, frame-by-frame tagging).

The `_` in the unpack is the `(h_n, c_n)` tuple, the final hidden and cell states across all layers. For a unidirectional LSTM with `batch_first=True`, `out[:, -1, :]` and `h_n[-1]` are literally the same vector. The code reaches for the output tensor instead of the discarded hidden state, which is the convention but slightly wasteful. Worth knowing both paths exist.

## batch_first is worth the argument

`nn.LSTM`'s default is `batch_first=False`, sequence-first, `(seq_len, batch, feature)`. Every other PyTorch layer expects batch-first. The default is a historical wart and nothing else. Setting `batch_first=True` on the layer and reshaping the input to match (`images.reshape(-1, sequence_length, input_size)`) is two lines of consistency you have to remember to keep aligned. Forget one, the model still trains, just on garbage axes. There is no traceback. You find out from a loss curve that never moves.

`num_layers=2` is the other quiet choice. It stacks two LSTM cells, the output of the first feeding the input of the second. The `num_layers` dimension in `h_0` and `c_0` plumbs the initialization through to both. Deeper recurrent stacks aren't free (they amplify the vanishing-gradient problem the LSTM gates are already fighting), but two is cheap insurance.

## What this file doesn't cover

Honest scope-setting, since this is one video's worth of code:

- **Two epochs.** Enough to confirm the loss decreases, nothing more. No best-checkpoint logic, no test-time evaluation beyond a single accuracy print at the end.
- **Unidirectional.** The model walks rows top to bottom and that's it. A digit is just as legible read upward, and `bidirectional=True` is a one-flag change worth understanding next.
- **No real sequence.** MNIST-as-sequence is a crutch. The actual point of recurrent nets is data where the ordering is meaningful: text, audio, time series. None of that is in this file yet.
- **No GRU comparison.** GRU is the smaller cousin of LSTM with one gate fewer and often comparable results. Worth a head-to-head later.
- **No gradient clipping.** Standard hygiene for recurrent training (`torch.nn.utils.clip_grad_norm_`) that the tutorial skipped because at 28 timesteps and 2 epochs it doesn't bite.

## Takeaway

Recurrent nets feel deliberately backwards after a month of convolutions. A CNN sees the whole image at once and lets the filters discover the structure. An LSTM walks across one slice at a time, with the gates deciding what to carry forward. The 28-row reshape bridges the two views on the same pixels. It's a useful warm-up but not a real application. The real application is the next video.
