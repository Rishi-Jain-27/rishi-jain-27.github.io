---
layout: post
title: "First PyTorch project, an MNIST classifier"
author: "Rishi Jain"
date: 2026-05-12
categories: projects
project_url: https://github.com/Rishi-Jain-27/mnist-classifier
project_label: Code
---

My first PyTorch project is a handwritten-digit classifier trained on MNIST. The dataset is a cliché, but that wasn't the point. I wanted to figure out what running ML experiments looks like end to end: comparing architectures honestly, logging metrics that survive a Colab session dying on you, and saving the right checkpoint when training spans 30 epochs and I'm not babysitting it. Code is on [GitHub](https://github.com/Rishi-Jain-27/mnist-classifier).

The thesis going in was that the model is the small part. The workflow is the thing that transfers.

## The architecture bake-off

Before writing one "real" model, I wanted a baseline I could trust. So I trained four architectures back to back on identical hyperparameters (10 epochs, `Adam(lr=0.001)`, `CrossEntropyLoss`, batch size 16, 128 hidden units) and let the numbers pick the winner.

| Model               | Layers                                |
| ------------------- | ------------------------------------- |
| Linear              | `Flatten → Linear → Linear → Linear`  |
| Non-Linear          | Linear + `GELU` activations           |
| Convolutional       | Two `Conv2d` blocks + `MaxPool2d`     |
| Conv + Non-Linear   | `Conv2d` + `GELU` + `MaxPool2d`       |

**Conv + Non-Linear won** on both accuracy and lowest test loss. The intuition is obvious in hindsight. Convolutions extract spatial features that a flat MLP can't see, and `GELU` lets the network model curved decision boundaries that a stack of linear layers can't. Knowing that in the abstract is not the same as running it. The point of the bake-off wasn't to discover that convolutions help on images. It was to build the habit of checking before committing to an architecture for the rest of the project.

## The real model

Once the bake-off told me where to invest, I built a deeper version called `MNISTClassifier`:

- Two convolutional blocks (`Conv2d → BatchNorm2d → GELU → MaxPool2d`, repeated).
- A classifier head with `Dropout(0.25)` and `Linear → GELU → Linear`.
- Trained 30 epochs with `SGD(lr=0.001, momentum=0.9, weight_decay=1e-4)`.
- The saved checkpoint is the one with the best test accuracy across the whole run, not the one from the last epoch.

That last detail mattered more than it sounds. On a 30-epoch run, the final-epoch model and the best-epoch model are almost never the same. Saving the wrong one quietly throws away your best result and you don't notice until you try to reproduce a number from your own logs.

## Infrastructure over architecture

The thing that took the most effort wasn't the model. It was [`helper_functions.py`](https://github.com/Rishi-Jain-27/mnist-classifier/blob/main/helper_functions.py), the part you reuse on every project after this one.

- **`train` / `train_step` / `test_step`.** The standard PyTorch loops, but `train` also writes scalars to TensorBoard and calls `save_best_model` every epoch, so checkpointing isn't something I have to remember to do.
- **`create_writer`.** Builds TensorBoard log dirs at `runs/YYYY-MM-DD/<experiment>/<model>/`. With that structure, comparing two experiments side by side is one `tensorboard --logdir runs` away.
- **`start_tensorboard_tunnel`.** Colab-friendly. It downloads `cloudflared`, kills stale processes, starts TensorBoard, and prints a public `trycloudflare.com` URL in a background thread. The first time it worked I felt like I'd cheated.
- **`set_seeds`.** Seeds Python, NumPy, PyTorch CPU and CUDA, and pins CuDNN to deterministic mode. A run that can't be repeated is barely a run.
- **`save_best_model`.** Takes a `mode="max"` (accuracy) or `mode="min"` (loss), compares against the best metric seen so far, and only writes the checkpoint if the current epoch improved on it.

None of this is glamorous and none of it shows up in a results table. It's the difference between training a model once and being able to run another experiment tomorrow and compare it to this one.

## What's next

The whole point of building this scaffolding was to have something to run more experiments on top of. On the list:

- Optimizer sweep: `Adam` vs `AdamW` vs `SGD`.
- Learning-rate tuning for whichever wins.
- `ReLU` vs `GELU` head to head on the same architecture.
- Batch size 16 vs 32.
- Data-augmentation transforms.
- Transfer learning from a pretrained backbone.

## Takeaway

MNIST is solved. That's part of why it's a good first project. With the dataset out of the way, what's left is the workflow: picking the model deliberately, logging everything, saving the best one rather than the last one, seeding so the result is reproducible, and making the tooling reusable so the next project starts ahead of where this one started.
