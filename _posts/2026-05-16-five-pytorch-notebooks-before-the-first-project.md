---
layout: post
title: "Five PyTorch notebooks I typed out by hand"
author: "Rishi Jain"
date: 2026-05-16
categories: projects
project_url: https://github.com/Rishi-Jain-27/pytorch-learning
project_label: Code
---

Before [the MNIST classifier]({% post_url 2026-05-12-mnist-classifier-first-pytorch-project %}) and [the FER2013 classifier]({% post_url 2026-05-14-fer2013-classifier-when-transfer-learning-loses %}), there were five Jupyter notebooks. They're homework, not a project, but the project posts that came after don't make much sense without them. The notebooks are on [GitHub](https://github.com/Rishi-Jain-27/pytorch-learning).

They are my pass through Daniel Bourke's *Learn PyTorch for Deep Learning* course, one notebook per chapter:

1. **Fundamentals.** Tensors and the three things that go wrong with them.
2. **Workflow.** The full data → model → train → save loop on a one-feature regression problem.
3. **Classification.** Binary, then multi-class, with the non-linearity reveal in the middle.
4. **Computer Vision.** FashionMNIST through a linear baseline, a non-linear baseline, and a CNN.
5. **Custom Datasets.** Loading images from disk, TinyVGG, and data augmentation.

About four weeks of evenings between April 15 and May 9.

## Typing, not watching

The method mattered more than the content. I typed every cell, including the imports, the print statements, and the cell-level comments I half-disagreed with, instead of cloning the repo and hitting Shift+Enter. That's slower by a factor of five. It is also the only way I personally retain anything technical.

Watching someone build a training loop is like watching someone parallel park. You nod along, the move looks obvious, and then you sit in the driver's seat and can't remember which way to turn the wheel. Typing the loop yourself is what gets it into your hands.

The notebooks have my fingerprints on them. Typos in comments, paraphrased explanations where the original phrasing didn't land, off-by-one notes to myself ("nomenclature: scalar and vector lowercase. matrix and tensor uppercase."). Those are the bits that survived past the last cell.

## The three things that go wrong

If I forget everything else from notebook 00, I want to keep this. Almost every PyTorch error in the first month is one of three things:

1. **Wrong datatype.** `torch.mean` on an int tensor. `float16` × `float32` getting silently promoted. Default dtype coming in as `float64` because the data went through NumPy first. `tensor.dtype`.
2. **Wrong shape.** A `(3, 2) @ (3, 2)` matmul that needs a transpose. A 4D image tensor where the model expects 3D. A `squeeze` you forgot. `tensor.shape`.
3. **Wrong device.** Tensor on CPU, model on GPU. A `.numpy()` call on a CUDA tensor. `tensor.device`.

This isn't deep but it's most of the skill of reading a PyTorch traceback for a while. Print all three. Find the mismatch. Move on.

## The inflection points

A few moments in the notebooks where the lights went on.

**Linear stacks can only fit lines.** Notebook 02 has you train a model with three `Linear` layers on a circles dataset. It plateaus at 50% accuracy, chance. You add a single `ReLU` between the layers and the decision boundary suddenly bends and the model hits 99%. Reading "non-linear activations let you fit non-linear functions" is one thing. Watching the boundary uncurl in matplotlib is another.

**Convolutions are just a smaller multiplication.** Notebook 03 walks `nn.Conv2d` and `nn.MaxPool2d` shape by shape with `(1, 1, 28, 28)` test inputs, and you can see the spatial dimension halve at each pool. There is no magic, just a kernel sliding across a tensor and a stride that decides how far it jumps. By the time `Conv2d` shows up in a real model in notebook 04, it's a known quantity.

**Best-epoch, not last-epoch.** Notebook 03 saves the model with the lowest test loss across the whole run, not whatever happened to be in memory when the loop exited. That single pattern became `save_best_model` in the MNIST project and carried straight into FER2013. It's two lines of code and it's the difference between "I trained this once" and "I trained this and saved the right one."

## What didn't stick the first time

Honest accounting: typing it out is not the same as understanding it. Things I had to come back and re-derive when the real projects started:

- **Logits → probabilities → labels.** I wrote the chain three times in notebook 02 and still applied softmax twice in my MNIST head the first time around. `CrossEntropyLoss` already applies it internally. You feed it raw logits. The notebook said this. I did not internalize it.
- **Channels-first vs. channels-last.** PyTorch wants `(N, C, H, W)`. Everything else (PIL, NumPy, matplotlib) wants `(H, W, C)`. The `permute(2, 0, 1)` trick is in the fundamentals notebook and I still hit a shape error on it in FER2013.
- **Class imbalance.** Not in any of the notebooks. The course assumes balanced datasets. The first real-world dataset I touched (FER2013) had a 16× imbalance between Happy and Disgust, and I had to learn `WeightedRandomSampler` from the docs on the fly.

The course doesn't claim to cover everything. The point is that it covers enough surface to make the gaps visible. You can't ask the right question on Stack Overflow if you don't know which thing is missing.

## Takeaway

The course gets a lot of skepticism online ("tutorials are a trap," "just build something"). That's right in spirit and wrong in practice for someone who has never written a training loop. You cannot build the thing if you do not know what shape it should be. Five notebooks of deliberate typing is what gave me the shape. The two project posts that came after are what filled it in.
