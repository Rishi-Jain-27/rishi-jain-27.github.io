---
layout: post
title: "Quantization notes, from FP32 down to packed 4-bit weights"
author: "Rishi Jain"
date: 2026-05-31
categories: notes
math: true
---

The other day I wanted to understand how a 7-billion-parameter model, which in full FP32 precision wants 28 GB just to hold its weights, gets squeezed onto a single consumer GPU. The trick is quantization: store each weight as an 8-bit integer rather than a 32-bit float and that 28 GB drops to 7 GB, with almost none of the accuracy lost along the way. These are my notes on how that works, worked through Zachary Huang's [Give me 30 min, I will make Quantization click forever](https://www.youtube.com/watch?v=SXeThN5S7u0), with the small numerical examples done by hand, a NumPy snippet for each section, and an interactive widget so you can watch the rounding actually happen. They follow the same format as my [linear algebra]({% post_url 2026-05-29-basic-linear-algebra-notes %}) and [eigenvector]({% post_url 2026-05-30-eigenvalue-notes %}) notes.

Quantization is the trip from float to int, and dequantization is the trip back to an approximation of the original float. All of the work is in making that approximation cheap to store while keeping it close enough that the model's outputs don't visibly degrade.

## What an FP32 float actually is

To see what we are compressing, it helps to look at what a float actually is. An IEEE-754 single-precision number splits its 32 bits into three fields. One sign bit \\( S \\) decides positive or negative, eight exponent bits \\( E \\) set the scale (how big or how tiny the number is), and the remaining twenty-three mantissa bits \\( M \\) hold the significant digits. Together they encode the value

\\[
x = (-1)^{S} \, (1 + M) \, 2^{\,E - 127}
\\]

Take \\( 3.5 \\). In binary that is \\( 11.1 \\), which normalizes to \\( 1.11 \times 2^{1} \\), so the sign is \\( 0 \\), the stored exponent is \\( 1 + 127 = 128 = 10000000_2 \\), and the mantissa is the fractional part \\( .11 \\) padded out to twenty-three bits as \\( 110\dots0 \\). One number, three fields, four bytes.

Those four bytes per number are the whole cost. A 7B model is \\( 7 \times 10^9 \times 4 \\) bytes \\( = 28\,\text{GB} \\) of weights before you have counted optimizer state or activations at all. INT8 throws the sign, exponent, and mantissa structure away entirely and keeps a single integer in \\( [-128, 127] \\), which is one byte, and so four times smaller.

## The affine quantization map

We have a continuous range of floats and we want to map every point in it onto one of a handful of discrete integer buckets. Two numbers define that map. The scale \\( S \\) is how much float-distance a single integer step is worth,

\\[
S = \frac{\text{float range}}{\text{integer range}}
\\]

and the zero-point \\( Z \\) is the integer that the float \\( 0.0 \\) lands on. If the weights are roughly centered on zero the map is symmetric and \\( Z = 0 \\), and if they are lopsided it is asymmetric with \\( Z \neq 0 \\). These notes stay in the symmetric \\( Z = 0 \\) case, which is the common one for weights and rather simpler to reason about.

With \\( Z = 0 \\), encoding is divide, round, and clamp,

\\[
q = \text{clamp}\left(\operatorname{round}\left(\frac{x}{S}\right),\; -128,\; 127\right)
\\]

and decoding is a single multiply,

\\[
\hat{x} = S \, q
\\]

The reconstructed \\( \hat{x} \\) lands on the nearest bucket center rather than on \\( x \\) itself, and the gap \\( \hat{x} - x \\) is the quantization error we agree to live with. For symmetric INT8 the scale is pinned to the largest magnitude in the tensor, mapping \\( \pm\lvert x \rvert\_{\max} \\) onto \\( \pm 127 \\):

\\[
S = \frac{\lvert x \rvert\_{\max}}{127}
\\]

## Working one tensor by hand

Take the weights from the video,

\\[
\mathbf{w} = [\,1.2,\; -3.5,\; 0.8,\; 2.1,\; -1.9,\; 3.5\,]
\\]

and work the map through by hand. The largest magnitude here is \\( 3.5 \\), so the scale is

\\[
S = \frac{3.5}{127} \approx 0.027559
\\]

To quantize \\( x = 1.2 \\) we divide by that scale and round,

\\[
\frac{1.2}{0.027559} \approx 43.54 \quad\longrightarrow\quad q = 44
\\]

and then to see the damage we dequantize and compare against the original,

\\[
\hat{x} = 44 \times 0.027559 \approx 1.2126
\\]

\\[
\text{error} = 1.2126 - 1.2000 = +0.0126
\\]

That is off by about one percent, which is the price of throwing away three of the four bytes. The whole tensor takes four lines in NumPy:

```python
import numpy as np

def symmetric_quantize(fp32_tensor):
    max_val = np.max(np.abs(fp32_tensor))
    scale = max_val / 127.0
    q = np.round(fp32_tensor / scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale
```

The widget below runs exactly this map. Each gold dot is a float weight, the diamond it drops onto is its dequantized value \\( \hat{x} \\), and the red stem between them is the error. You can flip between INT8, INT4, and INT3 to coarsen the grid (fewer levels, longer red stems), and drag a dot around to feel the rounding move under your hand.

{% include viz/quantize.html weights="1.2,-3.5,0.8,2.1,-1.9,3.5" bits="4" %}

## Weights versus inputs

A linear layer is just a matmul, \\( \text{output} = \text{input} \times \text{weights} \\), so a natural question is why we do not simply compress both sides of it. The answer is that the two operands behave very differently. The weights are learned knowledge: static, fixed at train time, sitting in VRAM, and huge, which is what makes them the memory bottleneck. The inputs are the user's prompt: dynamic, different on every single call, streaming through cache, and small. Because the weights are the bottleneck, compressing them is where the entire win lives, taking the model from 28 GB down to 7 GB, whereas quantizing the inputs would buy us almost nothing and cost a `quant()` / `dequant()` round-trip at every layer of every forward pass.

That raises a fair question of its own: if the weights are INT8 and the inputs are still FP16, how do we multiply them at all? We do not multiply them directly. On-the-fly dequantization expands a tile of weights back to float in the cache, nanoseconds before the matmul, runs the arithmetic in full precision, and then throws the float copy away. We pay the memory cost of INT8 while doing the actual computation in FP16, so the storage saving comes without a precision hit where it would matter.

## What one outlier does to the scale

The scale is pinned to \\( \lvert x \rvert\_{\max} \\), which quietly makes the whole scheme hostage to its single largest value. Swap the last weight for a spike and watch what happens,

\\[
\mathbf{w} = [\,1.2,\; -3.5,\; 0.8,\; 2.1,\; -1.9,\; 1000.0\,]
\\]

Now \\( \lvert x \rvert\_{\max} = 1000 \\), so the scale balloons to

\\[
S = \frac{1000}{127} \approx 7.87
\\]

A single step is now worth 7.87, and every normal weight is smaller than one step,

\\[
\operatorname{round}\left(\frac{1.2}{7.87}\right) = \operatorname{round}(0.15) = 0 \quad\longrightarrow\quad \hat{x} = 0
\\]

The \\( 1.2 \\) rounds away to nothing, and so does everything else in the normal range, since they all round to \\( 0 \\) and dequantize straight back to \\( 0 \\). One outlier stretched the grid so wide that the rest of the tensor simply fell through the cracks, which is a bit of a disaster for a layer that was meant to remember all six numbers. If you go back to the widget and hit "spike an outlier", the diamonds for the small weights all pile onto zero and `collisions` jumps.

## Granularity

The problem is using a single scale for the whole tensor, so the fix is to shrink the territory each scale is responsible for, until an outlier can only poison its own small neighborhood. This is granularity, and it comes in three increasingly fine settings. Per-tensor quantization uses one \\( S \\) for the entire matrix, which is the cheapest option and, as we just saw, the one where a single outlier ruins everything. Per-channel quantization uses one \\( S \\) per row, which is the INT8 standard, so an outlier corrupts only its own row and leaves the others clean. Group-wise quantization goes finer still, splitting each row into blocks of 64 or 128 and giving each block its own \\( S \\). This is the 4-bit standard, with the most isolation, paid for by storing a great many more scale factors.

{% include viz/granularity.html %}

Laid out on the same matrix the three settings come apart cleanly. Per-tensor draws one box around everything, so the blown-out scale reaches every cell. Per-channel cuts that box into rows, and the outlier can only spoil the row it actually sits in. Per-group cuts each row again into blocks, so the damage shrinks to the single block that holds the spike, with everything around it quantizing as if the outlier were never there. The cost is the mirror image of the benefit: one scale, then three, then six, and in a real tensor with blocks of 64 or 128 that scale count runs into the thousands.

Per-channel in NumPy is a single `axis` argument, taking the max along each row instead of over the whole array:

```python
import numpy as np

weights_fp32 = np.array([
    [ 1.2, -0.5,   2.8,  0.9],   # ok
    [-1.5, 1000.0, 0.3, -2.1],   # this row has the outlier
    [ 3.1, -2.2,  -1.8,  1.1],   # ok
], dtype=np.float32)

abs_max = np.max(np.abs(weights_fp32), axis=1, keepdims=True)
scales = abs_max / 127.0
print(scales.flatten())   # [0.022, 7.874, 0.024]
```

Only the middle scale is blown out. Rows 0 and 2 keep a fine \\( S \approx 0.022 \\) and quantize cleanly, while the damage stays quarantined to the row that actually contains the \\( 1000.0 \\). Granularity is really just a dial that trades scale-storage overhead against quality: per-tensor is the cheap baseline with the lowest quality, per-channel is the INT8 standard sitting in the middle on both counts, and group-wise buys the highest quality for the highest overhead.

## INT4 and the byte barrier

INT8 already cut FP16's 14 GB down to 7 GB. INT4 would halve it once more to 3.5 GB, a \\( 4\times \\) saving over FP16, and the map is the same affine one with a tiny range, \\( q \in [-8, 7] \\):

\\[
S = \frac{\lvert x \rvert\_{\max}}{7}
\\]

With only sixteen buckets to work with, group-wise quantization stops being a nice-to-have and becomes mandatory, because trying to fit a whole tensor into sixteen levels destroys it and we have no choice but to work in small isolated groups. Take one such group and watch two close values collide,

\\[
\text{group} = [\,0.51,\; 0.58,\; -1.2,\; 2.1\,], \qquad S = \frac{2.1}{7} = 0.3
\\]

\\[
\operatorname{round}\left(\frac{0.51}{0.3}\right) = 2, \qquad \operatorname{round}\left(\frac{0.58}{0.3}\right) = 2
\\]

Two distinct floats get crushed onto the same integer, and that collision is precisely the precision we trade away for the extra \\( 2\times \\). Switching the widget to INT4, or INT3, shows just how few diamonds are left for the dots to land on.

There is also a hardware wall waiting here. Memory is byte-addressable, so the smallest thing a CPU or GPU can load is eight bits rather than four, and a lone 4-bit value cannot physically sit in memory on its own.

## Packing two 4-bit values into a byte

The fix is to store two 4-bit values inside one 8-bit byte. We shift the second value into the high four bits and OR the first into the low four bits. If the two quantized values are \\( 5 = 0101_2 \\) and \\( 4 = 0100_2 \\), they pack together as

\\[
\underbrace{0100}\_{\text{high 4 bits}} \;\vert\; \underbrace{0101}\_{\text{low 4 bits}} \;=\; 01000101_2 = 69
\\]

```python
first, second = 5, 4
packed = (second << 4) | first    # 0100 0101 = 69

lo = packed & 0x0F                # 5
hi = (packed >> 4) & 0x0F         # 4
```

We pack on the way to disk, then unpack with a mask and a shift on the way into the matmul. The model now stores a genuine four bits per weight while still respecting the byte-addressable hardware underneath it.

## PTQ and QAT

There are two strategies, and which one you reach for depends on whether you quantize after training or during it.

Post-training quantization, PTQ, is the default for almost everyone. We start from the trained FP16 model, run a calibration pass of perhaps a hundred samples to observe the min and max ranges of the activations and pick good scales, then quantize and save. It takes minutes on a single GPU, needs no access to the full training set (which is a quiet privacy win), and with group-wise quantization the accuracy loss is close to negligible.

Quantization-aware training, QAT, is the heavier option, for when that residual loss is still too large, or when the application cannot tolerate even a minor drop in accuracy. During fine-tuning we insert fake quantization, where in the forward pass each weight is rounded to INT and immediately cast back to FP32, injecting the exact rounding noise the deployed model will eventually see. The weights are damaged slightly on purpose, and because this is happening inside a training loop, gradient descent adapts them to absorb that very damage.

{% include viz/ptq-vs-qat.html %}

The difference comes down to where the rounding sits relative to the training loop, which is what the diagram above is really about: PTQ waits until the weights are frozen and then quantizes once, whereas QAT folds the rounding into every forward pass so the optimizer sees it coming and compensates before anything is frozen.

Put plainly, PTQ is the low-effort path: you point it at a trained model and run it, it wants only a tiny calibration set and a few minutes of compute, and it covers the modern LLMs almost everyone is actually deploying, the Llamas and GPTs of the world. QAT is a full training pipeline that wants the entire training set and days of GPU-hours, and it earns its keep on edge and vision models, or when you are pushing all the way down to INT4.

## Takeaway

The arithmetic underneath all of this is just \\( q = \operatorname{round}(x / S) \\) on the way down and \\( \hat{x} = S q \\) on the way back up, and almost everything else is the engineering of choosing \\( S \\) so that one fat weight cannot drag the rest of the tensor to zero. Per-channel scales are enough to make INT8 behave, group-wise scales and 4-bit packing get you down to INT4, and PTQ quietly handles the large majority of real LLM deployments. QAT is the thing you reach for once you have actually measured a loss you cannot live with.
