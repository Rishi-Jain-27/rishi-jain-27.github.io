---
layout: post
title: "A Hundred Timesteps Is a Hundred Layers — Backprop Through Time, From Scratch"
author: "Rishi Jain"
date: 2026-05-16 12:00:00
categories: projects
math: true
---

A companion to [the first LSTM post]({% post_url 2026-05-16-rnn-learning-first-lstm-in-pytorch %}). That one was about wiring the API together; this one is about what the training algorithm is actually doing on the backward pass. Backpropagation Through Time is not a separate algorithm. It is standard backpropagation applied to a graph whose weight matrices happen to be shared across a long chain of repeated applications. The two consequences of that sharing — gradients summed over every use of the weight, and a Jacobian product whose effective depth equals the sequence length — are the entire reason RNN training has its own personality. The goal here is to derive both, then watch them produce vanishing gradients with real numbers.

## The Baseline: Standard Backprop

In a feedforward net, each layer has its own weight matrix \( W_l \). The forward pass computes

\[
z_l = W_l \, a_{l-1} + b_l, \qquad a_l = f(z_l).
\]

The backward pass is one application of the chain rule. Define the error signal \( \delta_l = \partial L / \partial z_l \). Then

\[
\delta_l = \left( W_{l+1}^\top \delta_{l+1} \right) \odot f'(z_l), \qquad \frac{\partial L}{\partial W_l} = \delta_l \, a_{l-1}^\top.
\]

One forward, one backward, one independent gradient per layer. \( W_1 \) and \( W_2 \) are different matrices that get independently updated. The depth of the unrolled chain is the depth of the model — typically 10 to 50 layers.

## Unrolling an RNN

A recurrent net has one recurrent weight matrix \( W \) (plus an input matrix \( U \), an output matrix \( V \), a bias \( b \)) that it reuses at every step. Forward at time \( t \):

\[
z_t = W h_{t-1} + U x_t + b, \qquad h_t = f(z_t),
\]

\[
y_t = V h_t, \qquad L_t = \ell(y_t, \text{target}_t), \qquad L = \sum_{t=1}^{T} L_t.
\]

(For many-to-one classification — like the MNIST-as-sequence model in the companion post — only \( L_T \) is non-zero. The sum collapses to a single term, but the math below doesn't change.)

Drawn on paper, this looks exactly like a feedforward net unrolled across \( T \) columns — except every column uses the *same* \( W \). That picture is BPTT in one diagram. The "through time" part is the unrolling. Everything else is standard backprop on the unrolled graph.

## Twist One: Shared Weights Mean Summed Gradients

Let \( W^{(k)} \) denote the copy of \( W \) used at step \( k \) in the unrolled graph. They are notationally distinct but numerically equal. The multivariate chain rule for shared parameters says that the gradient at \( W \) is the sum across every place \( W \) appears:

\[
\frac{\partial L}{\partial W} = \sum_{k=1}^{T} \frac{\partial L}{\partial W^{(k)}}.
\]

Each term tracks how much the loss changes when only \( W^{(k)} \) is perturbed. That perturbation enters at step \( k \) and propagates forward to affect every future loss \( L_t \) for \( t \ge k \):

\[
\frac{\partial L}{\partial W^{(k)}} = \sum_{t=k}^{T} \left( \frac{\partial L_t}{\partial h_t} \right)^\top \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W^{(k)}}.
\]

Combine and swap the order of summation:

\[
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \sum_{k=1}^{t} \left( \frac{\partial L_t}{\partial h_t} \right)^\top \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W^{(k)}}.
\]

Read that twice. The inner sum says: for the loss at step \( t \), accumulate contributions from \( W \)'s use at every earlier step \( k \le t \). The outer sum says: do that for every \( t \). In ordinary backprop each weight gets one gradient computation per backward pass. Here it gets one per \( (t, k) \) pair, all summed into the same matrix. That's the first difference, and it's the reason a small RNN with 50k parameters costs more per training step than a feedforward net with the same parameter count — parameter count and gradient work are no longer proportional.

## The Single-Step Jacobian

The middle term \( \partial h_t / \partial h_k \) is the heart of BPTT. By the chain rule it factors as a product of single-step Jacobians:

\[
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}.
\]

Each factor is a derivative we can compute directly. Differentiating \( h_i = f(W h_{i-1} + U x_i + b) \) with respect to \( h_{i-1} \), using that \( f \) is applied elementwise:

\[
\frac{\partial h_i}{\partial h_{i-1}} = \mathrm{diag}\!\left( f'(z_i) \right) \, W.
\]

(In the backward-pass view, the gradient flowing from \( h_t \) to \( h_k \) multiplies by the transpose of this product. Singular values are invariant under transpose, so the operator-norm analysis below is identical either way.)

## Twist Two: The Recurrent Jacobian Product

Substituting the single-step form:

\[
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \mathrm{diag}\!\left( f'(z_i) \right) \, W.
\]

Take the operator norm and use submultiplicativity:

\[
\left\| \frac{\partial h_t}{\partial h_k} \right\| \le \prod_{i=k+1}^{t} \left\| \mathrm{diag}\!\left( f'(z_i) \right) \right\| \cdot \left\| W \right\|.
\]

That product is the whole problem. The *same* \( W \) every step. If the largest singular value \( \sigma_1(W) \) times the saturating factor from \( f' \) is reliably less than 1, the product shrinks geometrically in the gap \( (t - k) \). A gradient from step 100 contributing back to step 0 has been multiplied through roughly 100 copies of a contractive matrix. It arrives at zero. Vanishing gradient.

If the product is reliably greater than 1, the same compounding runs the other direction. Exploding gradient.

A standard backprop chain has the same shape — \( \prod_l \mathrm{diag}(f'(z_l)) \, W_l \) — but every \( W_l \) is a different matrix. Their singular values don't compound as cleanly, and the depth is the model depth, not the sequence length. The math is identical; the brutality is not.

## A Worked Vanishing-Gradient Example

The bound becomes much more visceral with numbers. Take \( f = \tanh \), so \( |f'(z)| \le 1 \) with equality only at \( z = 0 \). In practice \( \tanh \) saturates fast, and on a trained recurrent net the typical entry of \( f'(z_i) \) sits around \( 0.5 \) or lower — the network *needs* the saturated regime to make non-trivial decisions.

Suppose the recurrent weight matrix has \( \sigma_1(W) = 0.9 \). The per-step Jacobian norm is bounded by

\[
\left\| \mathrm{diag}(f'(z_i)) \right\| \cdot \left\| W \right\| \le \underbrace{0.5}_{\text{typical } f'} \cdot \underbrace{0.9}_{\sigma_1(W)} = 0.45.
\]

After unrolling 100 steps,

\[
\left\| \frac{\partial h_{100}}{\partial h_0} \right\| \le 0.45^{100} \approx 1.6 \times 10^{-35}.
\]

That number is below `float32` precision (`~1.2e-38`) and well below `float16`. The gradient isn't small — it is *gone*. No floating-point system in standard use can represent it without underflow.

Flip the inequality the other way: \( \sigma_1(W) = 1.2 \), and assume the model has learned to keep \( f' \) near 1 in the operating regime. Then

\[
\left\| \frac{\partial h_{50}}{\partial h_0} \right\| \le 1.2^{50} \approx 9100,
\]

and the parameter update is large enough to throw the model across the loss landscape in a single step. Exploding gradient.

The whole personality of RNN training — gradient clipping, careful initialization of \( W \) to have spectral radius near 1, the obsession with LSTM and GRU gates — is a response to these two regimes living one numerical eyelash apart.

## Why Clipping Helps One Side and Not the Other

Gradient clipping rescales when the global gradient norm exceeds a threshold:

\[
g \leftarrow g \cdot \min\!\left(1, \frac{\tau}{\|g\|}\right).
\]

This is a rescaling, not a rotation — the direction is preserved, only the magnitude is capped. For the exploding case that's exactly what you want: there is plenty of gradient signal, you just have too much of it, and rescaling gives you a bounded-magnitude step that still points the right way.

It does nothing for vanishing gradients. Once a number has underflowed to zero — or to the smallest representable denormal — there is no information to rescale up. You can multiply zero by any constant and still get zero. Clipping is for the case where you have signal you can't use; vanishing is the case where the signal isn't there. That asymmetry is why "just clip" handles the explosion failure mode in practice, and architectural changes (LSTM/GRU gates, residual connections, careful initialization, attention) are what's needed for the vanishing one.

## Why LSTMs Were Invented

LSTMs do not replace BPTT. The training algorithm is still BPTT. What they change is the recurrence. The cell-state update

\[
c_t = f_t \odot c_{t-1} + i_t \odot g_t
\]

has, treating the gates as locally constant in \( c_{t-1} \),

\[
\frac{\partial c_t}{\partial c_{t-1}} \approx \mathrm{diag}(f_t).
\]

When the forget gate \( f_t \) stays near 1, this Jacobian is approximately the identity. The product across timesteps is then approximately

\[
\frac{\partial c_t}{\partial c_k} \approx \prod_{i=k+1}^{t} \mathrm{diag}(f_i),
\]

a product of diagonal matrices with entries bounded by 1 and (when the network has decided to keep information) close to 1. The exponential decay that wrecks a generic recurrence stops happening.

The honest caveat: the gates themselves are functions of \( h_{t-1} \), which is a function of \( c_{t-1} \) through the standard output equation. So the full Jacobian has additional off-diagonal terms from that indirect path. The "gradient highway" argument is about the *dominant* path, not the only one. But it is the dominant path because the additive structure of the cell state means the near-identity Jacobian survives even when the gate-feedback path doesn't. Without the additive shortcut you'd be back to the same multiplicative product, and the same exponential decay.

This is also why the original 1997 LSTM paper calls the cell state the "constant error carousel" — it's the structural feature that lets the gradient ride along the identity for many steps without being attenuated.

## Truncated BPTT

Storing every \( h_t \) across a full sequence so the backward pass can reach them is \( O(T \cdot d) \) in memory, where \( d \) is the hidden size. For \( T = 1000 \) and \( d = 512 \), that's half a million activations per training example just for the hidden states (plus intermediates). For long sequences and large batches it stops fitting.

Truncated BPTT picks a window \( k \) (commonly 32, 64, 128) and only backpropagates \( k \) steps. Anything older than \( k \) is detached from the computation graph. Formally, the truncated gradient drops the outer sum's range:

\[
\frac{\partial L}{\partial W} \;\approx\; \sum_{t=1}^{T} \sum_{k=\max(1,\, t-K)}^{t} \left( \frac{\partial L_t}{\partial h_t} \right)^\top \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W^{(k)}}.
\]

You lose the gradient contribution from dependencies longer than \( K \), but the unrolled depth and the activation memory are both bounded by \( K \). Truncation is a hyperparameter, not a fix — it's the explicit decision about how far back is worth caring about. If the answer is "further than \( K \)," you need either a larger \( K \) (more memory) or an architecture that lets gradient propagate without geometric decay (back to LSTM).

## Takeaway

Backprop Through Time is backprop. The chain rule doesn't change; what changes is that the parameter being differentiated against shows up many times in the computation graph, and that the graph is unrolled to whatever depth the sequence length forces. Every characteristic complaint about RNN training — vanishing gradients, exploding gradients, gradient clipping, LSTM and GRU gates, truncated windows, the difficulty of long-range dependencies — is downstream of two equations: the sum over every use of \( W \), and the product of single-step Jacobians \( \prod \mathrm{diag}(f'(z_i)) \, W \). Internalize those two and the rest of the recurrent-net curriculum stops looking like a grab-bag of tricks and starts looking like a sequence of reasonable responses to a single underlying problem. The problem is the product. Everything else is engineering around it.
