---
layout: post
title: "MNIST Was the Excuse — Learning to Run Real Experiments in PyTorch"
author: "Rishi Jain"
date: 2026-05-12
categories: projects
project_url: https://github.com/Rishi-Jain-27/mnist-classifier
project_label: Code
---

My first PyTorch project is a handwritten-digit classifier trained on MNIST. The dataset is a cliché; the project wasn't really about it. The point was to figure out what running ML experiments actually looks like end-to-end — comparing architectures honestly, logging metrics that survive a Colab session dying on you, and saving the *right* checkpoint when training spans 30 epochs and I'm not babysitting it. Code is on [GitHub](https://github.com/Rishi-Jain-27/mnist-classifier).

The thesis going in: the model is the small part. The workflow is the thing that transfers.

## The Architecture Bake-Off

Before writing one "real" model, I wanted a baseline I could trust. So I trained four architectures back-to-back on identical hyperparameters — 10 epochs, `Adam(lr=0.001)`, `CrossEntropyLoss`, batch size 16, 128 hidden units — and let the numbers pick the winner.

| Model               | Layers                                |
| ------------------- | ------------------------------------- |
| Linear              | `Flatten → Linear → Linear → Linear`  |
| Non-Linear          | Linear + `GELU` activations           |
| Convolutional       | Two `Conv2d` blocks + `MaxPool2d`     |
| Conv + Non-Linear   | `Conv2d` + `GELU` + `MaxPool2d`       |

**Conv + Non-Linear won** on both accuracy and lowest test loss. The intuition is obvious in hindsight: convolutions extract spatial features that a flat MLP can't see, and `GELU` lets the network model the curved decision boundaries a stack of linear layers just can't. But knowing that in the abstract is not the same as running it. The point of the bake-off wasn't to discover that convolutions help on images — it was to build the habit of *checking* before committing to an architecture for the rest of the project.

## The Real Model

Once the bake-off told me where to invest, I built a deeper version — `MNISTClassifier`:

- Two convolutional blocks (`Conv2d → GELU → MaxPool2d`, repeated).
- A classifier head with `Dropout(0.25)` and `Linear → GELU → Linear`.
- Trained 30 epochs with `SGD(lr=0.001, momentum=0.9, weight_decay=1e-4)`.
- The checkpoint that gets saved is automatically the one with the best test accuracy *across the entire run* — not the one from the last epoch.

That last detail mattered more than it sounds. On a 30-epoch run, the final-epoch model and the best-epoch model are almost never the same. Saving the wrong one quietly throws away your best result and you don't notice until you try to reproduce a number from your own logs.

## Infrastructure Over Architecture

The thing that actually took the most effort wasn't the model. It was [`helper_functions.py`](https://github.com/Rishi-Jain-27/mnist-classifier/blob/main/helper_functions.py) — the part you reuse on every project after this one.

- **`train` / `train_step` / `test_step`** — the standard PyTorch loops, but `train` also writes scalars to TensorBoard and calls `save_best_model` every epoch, so checkpointing isn't something I have to remember to do.
- **`create_writer`** — builds TensorBoard log dirs at `runs/YYYY-MM-DD/<experiment>/<model>/`. With that structure, comparing two experiments side-by-side is one `tensorboard --logdir runs` away.
- **`start_tensorboard_tunnel`** — Colab- and remote-friendly. It downloads `cloudflared`, kills stale processes, starts TensorBoard, and prints a public `trycloudflare.com` URL in a background thread. The first time it worked I felt like I'd cheated.
- **`set_seeds`** — seeds Python, NumPy, PyTorch CPU and CUDA, and pins CuDNN to deterministic mode. A run that can't be repeated is barely a run.
- **`save_best_model`** — takes a `mode="max"` (accuracy) or `mode="min"` (loss), compares against the best metric seen so far, and only writes the checkpoint if the current epoch actually improved.

None of this is glamorous. None of it shows up in a results table. It is, however, the difference between *"I trained a model once"* and *"I can run another experiment tomorrow and compare it to this one."*

## What's Next

The whole point of building this scaffolding was to have something to run more experiments on top of. On the list:

- Optimizer sweep — `Adam` vs `AdamW` vs `SGD`.
- Learning-rate tuning for whichever wins.
- `ReLU` vs `GELU` head-to-head on the same architecture.
- Batch size 16 vs 32.
- Data-augmentation transforms.
- Transfer learning from a pretrained backbone.

## Takeaway

MNIST is solved. That's exactly why it's a good first project — with the dataset out of the way, what's left is the workflow. Pick the model deliberately. Log everything. Save the best one, not the last one. Seed it so you can prove the result. Make the tooling reusable. The model goes in the trash; the infrastructure ships to every project after it.
