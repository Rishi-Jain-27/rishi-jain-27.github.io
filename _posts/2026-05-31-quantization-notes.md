---
layout: post
title: "Quantization notes, from FP32 down to packed 4-bit weights"
author: "Rishi Jain"
date: 2026-05-31
categories: notes
math: true
---

A 7-billion-parameter model in FP32 wants 28 GB just for its weights. Store each one as an 8-bit integer instead and that drops to 7 GB, with almost no accuracy lost. These are my notes on how that trick works, following Zachary Huang's [Give me 30 min, I will make Quantization click forever](https://www.youtube.com/watch?v=SXeThN5S7u0), worked out with the small numerical examples, a NumPy snippet per section, and an interactive widget so you can watch the rounding happen. Same format as my [linear algebra]({% post_url 2026-05-29-basic-linear-algebra-notes %}) and [eigenvector]({% post_url 2026-05-30-eigenvalue-notes %}) notes.

Quantization is the trip from float to int. Dequantization is the trip back, to an approximation of the original float. The whole game is making that approximation cheap to store and close enough to not matter.

## How FP32 spends its 32 bits

To see what we're compressing, look at what a float actually is. An IEEE-754 single-precision number splits 32 bits into three fields:

1. **1 sign bit** \\( S \\): positive or negative.
2. **8 exponent bits** \\( E \\): the scale, how big or how tiny.
3. **23 mantissa bits** \\( M \\): the significant digits.

The value they encode is

\\[
x = (-1)^{S} \, (1 + M) \, 2^{\,E - 127}
\\]

Take \\( 3.5 \\). In binary that's \\( 11.1 \\), normalized to \\( 1.11 \times 2^{1} \\). So the sign is \\( 0 \\), the stored exponent is \\( 1 + 127 = 128 = 10000000_2 \\), and the mantissa is the fractional part \\( .11 \\) padded out to 23 bits as \\( 110\dots0 \\). Three fields, four bytes, one number.

That is the problem in one line: four bytes per number. A 7B model is \\( 7 \times 10^9 \times 4 \\) bytes \\( = 28\,\text{GB} \\) of weights, before you count optimizer state or activations. INT8 throws away the sign/exponent/mantissa structure entirely and keeps a single integer in \\( [-128, 127] \\): one byte, four times smaller.

## The affine quantization map

We have a continuous range of floats and want to map every point in it to one of a few discrete integer buckets. Two numbers define the map.

The **scale** \\( S \\) is how much float-distance one integer step is worth:

\\[
S = \frac{\text{float range}}{\text{integer range}}
\\]

The **zero-point** \\( Z \\) is the integer that the float \\( 0.0 \\) lands on. If the weights are roughly centered on zero, the map is *symmetric* and \\( Z = 0 \\). If they're lopsided, it's *asymmetric* and \\( Z \neq 0 \\). These notes stay in the symmetric \\( Z = 0 \\) case, which is the common one for weights.

With \\( Z = 0 \\), encoding is divide-round-clamp:

\\[
q = \text{clamp}\!\left(\operatorname{round}\!\left(\frac{x}{S}\right),\; -128,\; 127\right)
\\]

and decoding is a single multiply:

\\[
\hat{x} = S \, q
\\]

\\( \hat{x} \\) is not \\( x \\). It's the nearest bucket center, and the gap \\( \hat{x} - x \\) is the quantization error we agree to live with.

For symmetric INT8 the scale is pinned to the largest magnitude in the tensor, mapping \\( \pm\lvert x \rvert\_{\max} \\) onto \\( \pm 127 \\):

\\[
S = \frac{\lvert x \rvert\_{\max}}{127}
\\]

## Working one tensor by hand

Take the weights from the video:

\\[
\mathbf{w} = [\,1.2,\; -3.5,\; 0.8,\; 2.1,\; -1.9,\; 3.5\,]
\\]

**1. Scale.** The largest magnitude is \\( 3.5 \\), so

\\[
S = \frac{3.5}{127} \approx 0.027559
\\]

**2. Quantize \\( x = 1.2 \\).** Divide by the scale and round:

\\[
\frac{1.2}{0.027559} \approx 43.54 \quad\longrightarrow\quad q = 44
\\]

**3. Dequantize and measure the damage.**

\\[
\hat{x} = 44 \times 0.027559 \approx 1.2126
\\]

\\[
\text{error} = 1.2126 - 1.2000 = +0.0126
\\]

Off by about one percent, from throwing away three of the four bytes. In code the whole tensor is four lines:

```python
import numpy as np

def symmetric_quantize(fp32_tensor):
    max_val = np.max(np.abs(fp32_tensor))      # scale is pinned to the peak
    scale = max_val / 127.0
    q = np.round(fp32_tensor / scale)
    q = np.clip(q, -128, 127).astype(np.int8)  # clamp into INT8 range
    return q, scale
```

The widget below runs exactly this map. Each gold dot is a float weight; the diamond it drops onto is its dequantized value \\( \hat{x} \\), and the red stem is the error. Flip between INT8, INT4, and INT3 to coarsen the grid (fewer levels, longer red stems), and drag a dot around to feel the rounding move.

{% include viz/quantize.html weights="1.2,-3.5,0.8,2.1,-1.9,3.5" bits="4" %}

## Quantize the weights, not the inputs

A linear layer is just a matmul, \\( \text{output} = \text{input} \times \text{weights} \\), so why not compress both sides? Because the two operands have opposite lives:

| | weights | inputs |
|---|---|---|
| what | learned knowledge | the user's prompt |
| when | static, set at train time | dynamic, different every call |
| where | sitting in VRAM | streaming through cache |
| size | huge, the bottleneck | small, transient |

Weights are the memory bottleneck, so compressing them is the whole win, 28 GB to 7 GB. Inputs are small and change on every forward pass, so quantizing them would mean paying for a `quant()` / `dequant()` round-trip at every layer of every inference. The standard move is weights-only.

That raises a question: if the weights are INT8 and the inputs are still FP16, how do you multiply them? You don't. On-the-fly dequantization expands a weight tile back to float in the cache, nanoseconds before the matmul, does the math in full precision, and discards the float copy. You pay the memory cost of INT8 and the compute in FP16: storage savings without a precision hit in the actual arithmetic.

## One outlier wrecks everything

The scale is pinned to \\( \lvert x \rvert\_{\max} \\), which makes the whole scheme hostage to its largest value. Swap the last weight for a spike:

\\[
\mathbf{w} = [\,1.2,\; -3.5,\; 0.8,\; 2.1,\; -1.9,\; 1000.0\,]
\\]

Now \\( \lvert x \rvert\_{\max} = 1000 \\), so

\\[
S = \frac{1000}{127} \approx 7.87
\\]

The step size is 7.87. Every normal weight is smaller than a single step:

\\[
\operatorname{round}\!\left(\frac{1.2}{7.87}\right) = \operatorname{round}(0.15) = 0 \quad\longrightarrow\quad \hat{x} = 0
\\]

The \\( 1.2 \\) is gone. So is everything else in the normal range; they all round to \\( 0 \\) and dequantize back to \\( 0 \\). One outlier stretched the grid so wide that the rest of the tensor fell through the cracks. Go back to the widget and hit **spike an outlier**: the diamonds for the small weights all pile onto zero, and `collisions` jumps.

## Granularity is the fix

The problem is using *one* scale for the whole tensor. Shrink the territory each scale is responsible for and an outlier can only poison its own neighborhood.

- **Per-tensor.** One \\( S \\) for the entire matrix. Cheapest, and one outlier ruins all of it.
- **Per-channel.** One \\( S \\) per row. This is the INT8 standard. An outlier corrupts its row and leaves the others clean.
- **Group-wise.** Split each row into blocks of 64 or 128 and give each block its own \\( S \\). This is the 4-bit standard. Maximum isolation, at the cost of storing many more scale factors.

Per-channel in NumPy is one `axis` argument, computing the max along each row instead of over the whole array:

```python
import numpy as np

weights_fp32 = np.array([
    [ 1.2, -0.5,   2.8,  0.9],   # ok
    [-1.5, 1000.0, 0.3, -2.1],   # this row has the outlier
    [ 3.1, -2.2,  -1.8,  1.1],   # ok
], dtype=np.float32)

# a separate scale per row (axis=1) instead of one global max
abs_max = np.max(np.abs(weights_fp32), axis=1, keepdims=True)
scales = abs_max / 127.0
print(scales.flatten())   # [0.022, 7.874, 0.024]
```

Only the middle scale is blown out. Rows 0 and 2 keep a fine \\( S \approx 0.022 \\) and quantize cleanly; the damage is quarantined to the row that actually contains the \\( 1000.0 \\). Granularity is the dial that trades scale-storage overhead for quality:

| method | standard for | quality | overhead |
|---|---|---|---|
| per-tensor | baseline | lowest | lowest |
| per-channel | INT8 | good | medium |
| group-wise | 4-bit | highest | highest |

## Halving it again: INT4 and the byte barrier

INT8 already cut FP16's 14 GB to 7 GB. INT4 would cut it again to 3.5 GB, a \\( 4\times \\) saving over FP16. The map is the same affine one with a tiny range, \\( q \in [-8, 7] \\):

\\[
S = \frac{\lvert x \rvert\_{\max}}{7}
\\]

Only sixteen buckets total, so group-wise quantization isn't optional here, it's mandatory. Trying to fit a whole tensor into 16 levels destroys it; you have to work in small isolated groups. Take one group and watch two close values collide:

\\[
\text{group} = [\,0.51,\; 0.58,\; -1.2,\; 2.1\,], \qquad S = \frac{2.1}{7} = 0.3
\\]

\\[
\operatorname{round}\!\left(\frac{0.51}{0.3}\right) = 2, \qquad \operatorname{round}\!\left(\frac{0.58}{0.3}\right) = 2
\\]

Two distinct floats crushed onto the same integer. That collision is the precision you trade away for the extra \\( 2\times \\). Switch the widget to INT4 (or INT3) to see how few diamonds are left for the dots to land on.

Then there's a hardware wall. Memory is byte-addressable; the smallest thing a CPU or GPU can load is 8 bits, not 4. A lone 4-bit value can't physically sit in memory on its own.

## Packing two nibbles into a byte

The fix is to stuff two 4-bit values into one 8-bit byte. Shift the second into the high nibble, OR the first into the low nibble. Say the two quantized values are \\( 5 = 0101_2 \\) and \\( 4 = 0100_2 \\):

\\[
\underbrace{0100}\_{\text{high nibble}} \;\vert\; \underbrace{0101}\_{\text{low nibble}} \;=\; 01000101_2 = 69
\\]

```python
first, second = 5, 4
packed = (second << 4) | first        # 0100 0101 = 69

lo = packed & 0x0F                     # 5, mask the low nibble
hi = (packed >> 4) & 0x0F              # 4, shift the high one down
```

Pack on the way to disk, unpack with a mask and a shift on the way into the matmul. The model is now genuinely 4 bits per weight, and the byte barrier never knows the difference.

## PTQ vs QAT: when to quantize

Two strategies, depending on whether you quantize after training or during it.

**Post-Training Quantization (PTQ)** is the default for almost everyone. Start from the trained FP16 model, run a *calibration* pass of maybe 100 samples to observe the min/max ranges of activations and pick good scales, quantize, save. It takes minutes on a single GPU, needs no access to the full training set (a privacy win), and with group-wise quantization the accuracy loss is close to negligible.

**Quantization-Aware Training (QAT)** is the heavy option for when that loss is still too big. During fine-tuning you insert *fake quantization*: in the forward pass each weight is rounded to INT and immediately cast back to FP32, injecting the exact rounding noise the deployed model will see. The weights are slightly damaged on purpose, and because it's inside a training loop, gradient descent adapts the weights to be robust to that damage.

| | PTQ | QAT |
|---|---|---|
| complexity | low, plug and play | high, full training pipeline |
| data needed | a tiny calibration set | the full training set |
| compute | minutes | days of GPU-hours |
| best for | modern LLMs (Llama, GPT) | edge/vision models, pushing to INT4 |

## Takeaway

The arithmetic of quantization is just \\( q = \operatorname{round}(x / S) \\) and \\( \hat{x} = S q \\); the engineering is all in choosing \\( S \\) so a single fat weight can't drag the rest of the tensor to zero. Per-channel scales handle INT8, group-wise scales and nibble-packing get you to INT4, and PTQ covers the large majority of real LLM deployments. Reach for QAT only when you've measured a loss you can't accept.
