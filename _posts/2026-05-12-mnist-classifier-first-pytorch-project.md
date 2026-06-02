---
layout: post
title: "First PyTorch project, an MNIST classifier"
author: "Rishi Jain"
date: 2026-05-12
categories: projects
project_url: https://github.com/Rishi-Jain-27/mnist-classifier
project_label: Code
---

My first PyTorch project is a handwritten-digit classifier trained on MNIST, which is about as much of a cliché as a first project can be, though the dataset was never really the point. What I actually wanted was to find out what running ML experiments looks like end to end, which means comparing architectures honestly rather than by vibe, logging metrics in a way that survives a Colab session dying on you halfway through, and saving the right checkpoint when training runs across 30 epochs and I am not sitting there babysitting it. The code is on [GitHub](https://github.com/Rishi-Jain-27/mnist-classifier), and the working assumption going in was that the model is the small part of all this and the workflow is the thing that carries over to everything after it.

## The architecture bake-off

Before writing a single "real" model I wanted a baseline I could actually trust, so I trained four architectures back to back on identical hyperparameters, the same 10 epochs and `Adam(lr=0.001)` and `CrossEntropyLoss` and batch size of 16 and 128 hidden units throughout, and let the numbers do the choosing rather than my intuition.

| Model               | Layers                                |
| ------------------- | ------------------------------------- |
| Linear              | `Flatten → Linear → Linear → Linear`  |
| Non-Linear          | Linear + `GELU` activations           |
| Convolutional       | Two `Conv2d` blocks + `MaxPool2d`     |
| Conv + Non-Linear   | `Conv2d` + `GELU` + `MaxPool2d`       |

The Conv + Non-Linear model won on both accuracy and lowest test loss, and the reason is obvious enough in hindsight, since the convolutions extract spatial features that a flat MLP simply cannot see while the `GELU` lets the network bend a decision boundary that a stack of linear layers can only ever keep straight. Knowing that in the abstract and watching it happen on your own run are quite different things, though, and the point of the bake-off was never to discover that convolutions help on images. It was to build the habit of checking before I commit to an architecture for the rest of a project, which is a habit that pays off long after MNIST is behind me.

## The real model

Once the bake-off had told me where the investment belonged, I built a deeper version called `MNISTClassifier`: two convolutional blocks, each a `Conv2d → BatchNorm2d → GELU → MaxPool2d` repeated, followed by a classifier head with `Dropout(0.25)` and a `Linear → GELU → Linear`, trained for 30 epochs with `SGD(lr=0.001, momentum=0.9, weight_decay=1e-4)`. The one detail in there that matters more than it looks is that the saved checkpoint is the one with the best test accuracy across the whole run rather than whatever happened to be sitting in memory at the final epoch. On a 30-epoch run the last-epoch model and the best-epoch model are almost never the same one, and saving the wrong one quietly throws your best result away without telling you, so that you only find out when you go to reproduce a number from your own logs and cannot.

## Infrastructure over architecture

The thing that took the most effort was not the model at all but [`helper_functions.py`](https://github.com/Rishi-Jain-27/mnist-classifier/blob/main/helper_functions.py), which is the part you actually reuse on every project after this one. The `train`, `train_step`, and `test_step` functions are the standard PyTorch loops, except that `train` also writes scalars to TensorBoard and calls `save_best_model` every epoch, so checkpointing stops being something I have to remember and becomes something that simply happens. `create_writer` builds the TensorBoard log directories under `runs/YYYY-MM-DD/<experiment>/<model>/`, and with that structure in place comparing two experiments side by side is one `tensorboard --logdir runs` away. `start_tensorboard_tunnel` is the Colab-friendly piece, which downloads `cloudflared`, kills any stale processes, starts TensorBoard, and prints a public `trycloudflare.com` URL from a background thread, and the first time it worked I felt as though I had cheated somehow. `set_seeds` seeds Python and NumPy and PyTorch on both CPU and CUDA and pins CuDNN to deterministic mode, because a run you cannot repeat is barely a run. And `save_best_model` takes a `mode` of either `max` for accuracy or `min` for loss, compares the current epoch against the best metric seen so far, and only writes the checkpoint when this epoch actually improved on it.

None of that is glamorous and none of it ever shows up in a results table, which is rather the point. It is the difference between training a model once and being able to come back tomorrow, run another experiment, and compare it honestly against this one.

## What's next

The whole reason for building this scaffolding was to have something solid to run more experiments on top of, and the list of those experiments is long enough already. There is an optimiser sweep to do, `Adam` against `AdamW` against `SGD`, and then learning-rate tuning for whichever of them wins, a `ReLU` against `GELU` comparison on the same architecture, a batch size of 16 against 32, a pass through some data-augmentation transforms, and eventually transfer learning from a pretrained backbone. Each of those is one experiment the scaffolding now makes cheap to run.

## Takeaway

MNIST is a solved dataset, and that is a large part of why it makes a good first project. With the dataset itself out of the way, what is left is exactly the workflow: choosing the model deliberately, logging everything, saving the best one rather than the last one, seeding so the result actually reproduces, and writing the tooling so the next project starts ahead of where this one began.
