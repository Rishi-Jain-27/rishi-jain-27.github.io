---
layout: post
title: "Backprop through time, from scratch"
author: "Rishi Jain"
date: 2026-05-16 12:00:00
categories: projects
math: true
---

The [first LSTM post]({% post_url 2026-05-16-rnn-learning-first-lstm-in-pytorch %}) was mostly a tour of the PyTorch API, the row-by-row reshape and the tuple-shaped hidden state and which line to keep for a many-to-one head, and it left the actual training algorithm sitting unexplained underneath the whole thing, so these are the notes that go after the backward pass properly. It assumes you are comfortable with the chain rule and with ordinary backprop, and builds from there. Backprop through time is standard backpropagation applied to a graph whose weight matrices happen to be shared across a long chain of repeated applications, and that single fact is most of the story. Two consequences fall straight out of that sharing, a gradient that sums over every use of the weight and a Jacobian product whose effective depth is the sequence length, and between them they are the entire reason RNN training has the personality it does. The plan is to derive both of them and then watch them produce a vanishing gradient with real numbers.

## The baseline: standard backprop

It helps to start from the case with no sharing in it at all. In an ordinary feedforward net each layer carries its own weight matrix \\( W_l \\), and the forward pass computes

\\[
z_l = W_l \, a_{l-1} + b_l, \qquad a_l = f(z_l).
\\]

The backward pass is a single application of the chain rule. Writing the error signal at a layer as \\( \delta_l = \partial L / \partial z_l \\), we have

\\[
\delta_l = \left( W_{l+1}^\top \delta_{l+1} \right) \odot f^\prime(z_l), \qquad \frac{\partial L}{\partial W_l} = \delta_l \, a_{l-1}^\top.
\\]

That is one forward pass, one backward pass, and one independent gradient for each layer, with \\( W_1 \\) and \\( W_2 \\) being genuinely different matrices that get updated on their own. The depth of the unrolled chain here is simply the depth of the model, somewhere around 10 to 50 layers in practice.

## Unrolling an RNN

A recurrent net has a single recurrent weight matrix \\( W \\), along with an input matrix \\( U \\), an output matrix \\( V \\), and a bias \\( b \\), and it reuses that same \\( W \\) at every step. The forward pass at time \\( t \\) is

\\[
z_t = W h_{t-1} + U x_t + b, \qquad h_t = f(z_t),
\\]

\\[
y_t = V h_t, \qquad L_t = \ell(y_t, \text{target}\_t), \qquad L = \sum_{t=1}^{T} L_t.
\\]

For many-to-one classification, like the MNIST-as-sequence model in the companion post, only \\( L_T \\) is non-zero, so the sum collapses to a single term, though none of the math below changes when it does. Drawn out on paper, this looks exactly like a feedforward net unrolled across \\( T \\) columns, with the one crucial difference that every column uses the same \\( W \\). That picture is the whole of BPTT in a single diagram: the unrolling is the "through time" part, and everything else is just standard backprop on the unrolled graph.

{% include viz/unrolled-weights.html %}

## Twist one: shared weights mean summed gradients

Let \\( W^{(k)} \\) denote the copy of \\( W \\) used at step \\( k \\) in the unrolled graph, where the copies are notationally distinct but numerically equal. The multivariate chain rule for shared parameters says the gradient at \\( W \\) is the sum across every place \\( W \\) appears:

\\[
\frac{\partial L}{\partial W} = \sum_{k=1}^{T} \frac{\partial L}{\partial W^{(k)}}.
\\]

Each term measures how much the loss changes when only \\( W^{(k)} \\) is perturbed, and that perturbation enters the graph at step \\( k \\) and then propagates forward to affect every future loss \\( L_t \\) for \\( t \ge k \\):

\\[
\frac{\partial L}{\partial W^{(k)}} = \sum_{t=k}^{T} \left( \frac{\partial L_t}{\partial h_t} \right)^\top \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W^{(k)}}.
\\]

Combining the two and swapping the order of summation gives

\\[
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \sum_{k=1}^{t} \left( \frac{\partial L_t}{\partial h_t} \right)^\top \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W^{(k)}}.
\\]

The inner sum says that for the loss at step \\( t \\) you accumulate contributions from \\( W \\)'s use at every earlier step \\( k \le t \\), and the outer sum says you then do that for every \\( t \\). Where ordinary backprop gives each weight one gradient computation per backward pass, here a single matrix collects one contribution per \\( (t, k) \\) pair, all summed into the same place. This is the first reason a small RNN with 50k parameters costs more per training step than a feedforward net with the same parameter count, since parameter count and gradient work have stopped being proportional to one another.

## The single-step Jacobian

The middle term \\( \partial h_t / \partial h_k \\) is the heart of the whole thing. By the chain rule it factors into a product of single-step Jacobians:

\\[
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}.
\\]

Each factor is a derivative we can write down directly. Differentiating \\( h_i = f(W h_{i-1} + U x_i + b) \\) with respect to \\( h_{i-1} \\), and using that \\( f \\) is applied elementwise, gives

\\[
\frac{\partial h_i}{\partial h_{i-1}} = \mathrm{diag}\\!\left( f^\prime(z_i) \right) \, W.
\\]

In the backward-pass view the gradient flowing from \\( h_t \\) back to \\( h_k \\) multiplies by the transpose of this product, and since singular values are invariant under transpose the operator-norm analysis below comes out identical either way.

## Twist two: the recurrent Jacobian product

Substituting the single-step form into the product gives

\\[
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \mathrm{diag}\\!\left( f^\prime(z_i) \right) \, W.
\\]

Taking the operator norm and using submultiplicativity,

\\[
\left\\| \frac{\partial h_t}{\partial h_k} \right\\| \le \prod_{i=k+1}^{t} \left\\| \mathrm{diag}\\!\left( f^\prime(z_i) \right) \right\\| \cdot \left\\| W \right\\|.
\\]

That product is the entire problem, because it is the same \\( W \\) at every step. If the largest singular value \\( \sigma_1(W) \\) times the saturating factor from \\( f^\prime \\) is reliably less than 1, the product shrinks geometrically in the gap \\( (t - k) \\), so that a gradient from step 100 contributing back to step 0 has been multiplied through roughly 100 copies of a contractive matrix and arrives at zero, which is the vanishing gradient. If the same product is reliably greater than 1, the compounding runs the other way and the gradient explodes instead. A standard backprop chain has the very same shape, \\( \prod_l \mathrm{diag}(f^\prime(z_l)) \, W_l \\), but there every \\( W_l \\) is a different matrix, so their singular values do not compound as cleanly and the depth is the model depth rather than the sequence length. The math is identical across the two cases, and only the recurrent one is brutal about it.

## A worked vanishing-gradient example

The bound gets a good deal more visceral with numbers in it. Take \\( f = \tanh \\), so that \\( \lvert f^\prime(z) \rvert \le 1 \\) with equality only at \\( z = 0 \\). In practice \\( \tanh \\) saturates fast, and on a trained recurrent net the typical entry of \\( f^\prime(z_i) \\) sits around 0.5 or lower, because the network needs the saturated regime in order to make any non-trivial decisions at all. Suppose the recurrent weight matrix has \\( \sigma_1(W) = 0.9 \\). Then the per-step Jacobian norm is bounded by

\\[
\left\\| \mathrm{diag}(f^\prime(z_i)) \right\\| \cdot \left\\| W \right\\| \le \underbrace{0.5}\_{\text{typical } f^\prime} \cdot \underbrace{0.9}\_{\sigma_1(W)} = 0.45.
\\]

After unrolling 100 steps that becomes

\\[
\left\\| \frac{\partial h_{100}}{\partial h_0} \right\\| \le 0.45^{100} \approx 1.6 \times 10^{-35}.
\\]

That number sits below `float32` precision (around `1.2e-38`) and well below `float16`, so the gradient has not merely become small, it has underflowed to zero, and no floating-point system in standard use can represent it without losing it entirely. Flip the inequality the other way, with \\( \sigma_1(W) = 1.2 \\) and a model that has learned to keep \\( f^\prime \\) near 1 in its operating regime, and you get

\\[
\left\\| \frac{\partial h_{50}}{\partial h_0} \right\\| \le 1.2^{50} \approx 9100,
\\]

which is large enough to throw the model clean across the loss landscape in a single update, the exploding gradient. The whole personality of RNN training, the gradient clipping and the careful initialization of \\( W \\) to a spectral radius near 1 and the obsession with LSTM and GRU gates, is a response to these two regimes living one numerical eyelash apart.

The widget below runs exactly this geometric product. The knob along the top sets the per-step factor \\( \gamma = \sigma_1(W) \cdot \lvert f^\prime \rvert \\), and the curve plots the bound \\( \gamma^n \\) across the steps back through time on a log axis, with the `float32` floor drawn in. Drag the knob below 1 and the curve dives through the floor and the gradient is gone after a few dozen steps, leave it at 1 and it rides flat, and push it above 1 and it climbs off the top of the chart.

{% include viz/gradient-through-time.html %}

## Why clipping helps one side and not the other

Gradient clipping rescales the gradient whenever its global norm exceeds a threshold:

\\[
g \leftarrow g \cdot \min\\!\left(1, \frac{\tau}{\\|g\\|}\right).
\\]

This is a rescaling rather than a rotation, so the direction is preserved and only the magnitude is capped, and for the exploding case that is precisely what you want, since you have plenty of gradient signal and simply too much of it, and rescaling hands you back a bounded-magnitude step that still points the right way. It does nothing whatever for vanishing gradients, because once a number has underflowed to zero, or to the smallest representable denormal, there is no information left in it to rescale upward, and multiplying zero by any constant is still zero. That is the asymmetry: clipping addresses the case where you have signal you cannot use, and the vanishing case is the one where the signal is simply not there, so clipping handles the explosion in practice while the architectural changes, the LSTM and GRU gates and residual connections and careful initialization and attention, are what is needed for the vanishing side.

## Why LSTMs were invented

LSTMs do not replace BPTT, since the training algorithm is still BPTT, and what they change instead is the recurrence itself. The cell-state update

\\[
c_t = f_t \odot c_{t-1} + i_t \odot g_t
\\]

has, treating the gates as locally constant in \\( c_{t-1} \\),

\\[
\frac{\partial c_t}{\partial c_{t-1}} \approx \mathrm{diag}(f_t).
\\]

When the forget gate \\( f_t \\) stays near 1 this Jacobian is approximately the identity, and the product across timesteps becomes

\\[
\frac{\partial c_t}{\partial c_k} \approx \prod_{i=k+1}^{t} \mathrm{diag}(f_i),
\\]

a product of diagonal matrices whose entries are bounded by 1 and, when the network has decided to hold onto information, close to 1, so that the exponential decay that wrecks a generic recurrence simply stops happening. The honest caveat is that the gates are themselves functions of \\( h_{t-1} \\), which depends on \\( c_{t-1} \\) through the standard output equation, so the full Jacobian carries additional off-diagonal terms from that indirect path, and the gradient-highway argument is really about the dominant path rather than the only one. It is the dominant path because the additive structure of the cell state lets the near-identity Jacobian survive even where the gate-feedback path does not, and without that additive shortcut you would be straight back to the same multiplicative product and the same exponential decay. This is also why the original 1997 [LSTM paper](https://www.bioinf.jku.at/publications/older/2604.pdf) calls the cell state the constant error carousel, since it is the structural feature that lets the gradient ride along the identity for many steps without being attenuated.

## Truncated BPTT

Storing every \\( h_t \\) across a full sequence, so that the backward pass can reach back to them, costs \\( O(T \cdot d) \\) in memory, where \\( d \\) is the hidden size. For \\( T = 1000 \\) and \\( d = 512 \\) that is half a million activations per training example just for the hidden states, before any of the intermediates, and for long sequences and large batches it stops fitting at all. Truncated BPTT picks a window \\( K \\), commonly 32 or 64 or 128, and only backpropagates \\( K \\) steps, detaching anything older than \\( K \\) from the computation graph, which formally drops the range of the outer sum to

\\[
\frac{\partial L}{\partial W} \;\approx\; \sum_{t=1}^{T} \sum_{k=\max(1,\, t-K)}^{t} \left( \frac{\partial L_t}{\partial h_t} \right)^\top \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W^{(k)}}.
\\]

You lose the gradient contribution from any dependency longer than \\( K \\), and in exchange the unrolled depth and the activation memory are both bounded by \\( K \\). Truncation is a hyperparameter rather than a fix, an explicit decision about how far back is worth caring about, and if the honest answer is "further than \\( K \\)," then you need either a larger \\( K \\) at the cost of more memory or an architecture that lets the gradient propagate without geometric decay, which sends you back to the LSTM.

## Takeaway

Backprop through time is backprop, and the chain rule does not change anywhere in it. What changes is that the parameter being differentiated against shows up many times over in the computation graph, and that the graph is unrolled to whatever depth the sequence length forces on it. Every characteristic complaint about RNN training, the vanishing and exploding gradients and the clipping and the LSTM and GRU gates and the truncated windows and the difficulty of long-range dependencies, is downstream of two equations, the sum over every use of \\( W \\) and the product of single-step Jacobians \\( \prod \mathrm{diag}(f^\prime(z_i)) \, W \\). With those two in hand the rest of the recurrent-net curriculum stops reading like a grab-bag of tricks and starts reading like a sequence of reasonable responses to the same underlying product.
