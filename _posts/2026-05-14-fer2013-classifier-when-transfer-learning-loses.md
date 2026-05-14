---
layout: post
title: "FER2013 — When Transfer Learning Loses to a Smaller Model"
author: "Rishi Jain"
date: 2026-05-14
categories: projects
project_url: https://github.com/Rishi-Jain-27/fer2013-classifier
project_label: Code
---

For my second PyTorch project I wanted something harder than MNIST. FER2013 fit the bill — 35,887 grayscale 48×48 face crops, sorted into seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). It's a famously rough dataset: human accuracy tops out around 65%, and published SOTA sits in the low 70s. I wasn't trying to beat SOTA. I wanted to take the infrastructure I'd built for the [MNIST classifier](https://github.com/Rishi-Jain-27/mnist-classifier) into a problem that would actually push back, and to check an assumption I'd absorbed somewhere along the way — that transfer learning is always the easy win. Code is on [GitHub](https://github.com/Rishi-Jain-27/fer2013-classifier).

The headline result, before I get to the how: my hand-rolled CNN beat a frozen VGG16 backbone, and the gap wasn't small.

## The Imbalance Problem

FER2013 is lopsided. There are around 7,200 training images of Happy faces and 436 of Disgust. Train a model naively on that and it learns to be confidently happy at everything — and worse, the accuracy number on paper looks reasonable because Happy dominates the distribution. The model gets to coast.

I corrected for this in two places at once. At the dataloader, a `WeightedRandomSampler` weighted each sample by the inverse of its class frequency, so every batch came out roughly balanced regardless of the underlying ratio. Then, for the transfer-learning model, I also passed class weights into `CrossEntropyLoss` so a missed Disgust example would cost more than a missed Happy one. Belt and suspenders, but with a 16× imbalance the default behavior is bad enough that the redundancy felt earned.

## The Self-Made Model

`FER2013V1` is three convolutional blocks and a small classifier head. Each block is the same shape — a pair of `Conv2d → BatchNorm → GELU` layers followed by `MaxPool(2)` — and the head is `Flatten → Dropout(0.25) → Linear(2304, 64) → GELU → Dropout(0.25) → Linear(64, 64) → GELU → Linear(64, 7)`.

The one unusual choice was the channel counts. Textbook CNNs widen as they go: 64 → 128 → 256. I went the other way, 256 → 128 → 64. The reasoning was that a 48×48 input doesn't have a lot of spatial structure to expand into. What it has is fine-grained pixel patterns — the curve of a mouth, the angle of a brow — and those benefit from a wide early layer that captures the texture before the maxpools start compressing it down. By the time the feature map is 6×6, 64 channels is plenty to describe what's left. It's a choice that only makes sense for genuinely tiny inputs.

I trained it with `AdamW(lr=0.001, weight_decay=1e-2)` and `CrossEntropyLoss`. About 1.3M parameters total.

## Transfer Learning Was Supposed to Be the Easy Win

The second model was the textbook recipe. Take VGG16 pretrained on ImageNet, freeze the feature extractor, swap on a new 7-class head, fine-tune. ImageNet has 14 million images in it. VGG16's filters had to be better than anything I could train from scratch on 28k tiny grayscale faces. That was the assumption, anyway.

Making it actually run took some massaging on the input side. Convert grayscale to 3-channel. Upscale 48×48 to 64×64 — smaller than VGG's native 224 to keep training tractable, but big enough that the conv stack wouldn't collapse to a single pixel. Apply ImageNet's normalization stats. Replace the 1000-class head with `Linear(25088, 512) → BatchNorm → Dropout(0.5) → Linear(512, 256) → BatchNorm → Dropout(0.3) → Linear(256, 7)`. Add a `ReduceLROnPlateau` scheduler, since I expected long, slow fine-tuning to be the move.

I trained it for 50 epochs — about 36 minutes on a Colab GPU — and the best test accuracy was 44.7%, hit at epoch 32 and never improved after.

Meanwhile the self-made CNN had cleared 49% by epoch 2 before my Colab session disconnected on me.

## Why the Bigger Model Lost

I was surprised by the result for about an afternoon, and then it made complete sense.

VGG16's early filters are tuned for natural-image statistics — color edges, ImageNet textures, the kind of object-scale gradients you get in a photograph of a dog. None of that is what separates "sad" from "neutral" in a 48×48 grayscale face. The signal there lives in small geometric differences around the eyes and mouth, and the pretrained filters have no particular reason to be sensitive to it.

The resolution made it worse. Even after upscaling, 64×64 is more than three times smaller than what VGG was trained on, and most of the network ends up processing feature maps that have already collapsed to 2×2 by the final block. There's almost no spatial information left for the classifier to work with.

And because the backbone is frozen, there's no way for the network to adapt. Train loss kept dropping — that's the head memorizing whatever signal does make it through — but test loss never followed. Classic head-overfitting on a base network that's locked into the wrong inductive bias.

It's the thing transfer-learning tutorials never quite say out loud: pretrained features only help when your problem looks like the pretraining task. 48×48 grayscale faces don't look like 224×224 ImageNet photos. A small purpose-built network beats a giant misapplied one.

## What Carried Over from MNIST

The entire [`helper_functions.py`](https://github.com/Rishi-Jain-27/fer2013-classifier/blob/main/helper_functions.py) module from the MNIST project dropped in mostly unchanged — `train`, `set_seeds`, `save_best_model`, `create_writer`, the Cloudflare-tunnel TensorBoard helper. That was the thesis of the last post: the model is the small part, the workflow is the thing that transfers. Two projects in, it's still holding up.

The only extension I made was adding an optional `scheduler` argument to `train` so `ReduceLROnPlateau` could step on the test loss each epoch. Small change. Worth keeping.

## What's Next

The self-made model is the right base; the next round is about pushing it further rather than swapping it out.

- Finish a clean 50-epoch run of the self-made model end-to-end. The disconnect cut me off at epoch 2.
- Plot a confusion matrix. I suspect Disgust and Fear are bleeding into each other even with the sampler doing its job.
- Try unfreezing the last block of VGG instead of freezing the whole feature extractor, and see whether partial fine-tuning closes the gap.
- Swap the plain conv blocks for a ResNet-style residual block, written from scratch.
- Tune data augmentation for faces specifically — small rotations and horizontal flips, no vertical flip. A person upside-down is not a person.

## Takeaway

The cliché going in was "always start with transfer learning." The result going out is more nuanced. Pretrained features help when the source and target tasks share an inductive bias, and they don't really otherwise. A network designed for the actual problem and trained from scratch on the actual data can beat a much larger pretrained one that was never going to fit the shape of what you're trying to do. Bigger isn't smarter. Closer is.
