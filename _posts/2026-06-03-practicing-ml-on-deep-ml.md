---
layout: post
title: "Practicing ML on Deep-ML, and fixing one of its problems"
author: "Rishi Jain"
date: 2026-06-03
categories: projects
project_url: https://www.deep-ml.com/
project_label: Visit
math: true
---

After a few months of working through PyTorch notebooks and first projects I wanted somewhere to practice the smaller pieces in isolation, the kind of thing that hides inside a library call and that you never really implement yourself: a dot product written out by hand, k-fold cross-validation, PCA, a decision tree split. The site I kept coming back to is [Deep-ML](https://www.deep-ml.com/), which is a LeetCode-shaped collection of machine-learning problems where you write a Python function in the browser and it runs against a set of test cases, except that the problems are the actual mechanics of ML rather than two-pointer puzzles dressed up in ML vocabulary. I ended up liking it enough to record a handful of walkthrough videos for it, and then liking it enough to go fix one of the problems when I found it was wrong, which is the part of this post I find more interesting than the recommendation itself.

## What the site is

The shape is familiar enough that there is not much to explain: you browse problems by category and difficulty, you write a solution in a browser editor, and you get instant feedback against the hidden tests. The categories are the ones you would want, the fundamentals first (regression, classification, clustering, dimensionality reduction) and then deep networks, computer vision, and NLP further up the difficulty curve, and the early problems are deliberately small enough that you implement the thing from scratch in a dozen lines rather than reaching for `numpy.linalg` or scikit-learn and calling it understood. That from-scratch constraint is the whole value of it for me, since the gap between knowing what PCA does and being able to write the covariance-eigendecomposition yourself is exactly the gap these problems sit in, and it is a gap that no amount of reading closes.

What makes it worth a post rather than a bookmark is that the problems double as a reason to implement the canonical algorithms one at a time, in the cleanest possible setting, with a test case telling you the moment you are wrong. I recorded a few of these as walkthroughs: the very first problem, a [matrix-vector dot product](https://youtu.be/uEV71j-dg7I), which is as basic as it sounds and is the right place to start, then [k-fold cross-validation](https://youtu.be/Zr0hV8fr5Fk), [PCA](https://youtu.be/gCc2VFMkk1s), and a [decision tree](https://youtu.be/ASX0YHqt6ns) built around information gain. The videos are me solving them out loud, which is a different exercise from solving them quietly, because narrating a derivation is unforgiving about the steps you were quietly skipping.

## The open-source part is the good part

The detail that turns Deep-ML from a decent practice site into something I would actively push people toward is that it is open source. The problems live in a public [GitHub repository](https://github.com/Open-Deep-ML/DML-OpenProblem), each one a small folder of Markdown for the description, a Python reference solution, and a JSON file of test cases, all assembled into the site by a build script. Anyone can read how a problem is graded, and more to the point anyone can fix a problem that is graded wrong, which on a community-contributed set of a few hundred problems is going to happen more than once.

This matters because the usual failure mode of a practice site is that a broken problem is a dead end: the tests reject your correct answer, you cannot see why, and you either contort your solution to match the bug or you give up and assume you are the one who is confused. An open repository turns that dead end into the most useful exercise on the whole site, because chasing down why a reference solution disagrees with the algorithm it claims to implement is a harder and more honest test of understanding than solving the problem ever was. You only catch the bug if you understand the algorithm better than the person who wrote the bug did, and there is no faking that.

## The Pegasos problem

The one I fixed is the kernelized Pegasos SVM, problem 21. Pegasos is a stochastic sub-gradient method for the soft-margin SVM, and the kernel version keeps a coefficient \\( \alpha_i \\) for each training point and predicts with the decision function

\\[ f(x_i) = \sum_j \alpha_j y_j K(x_j, x_i) + b \\]

where \\( K \\) is the kernel, \\( y_j \\) the label, and the step size \\( \eta = \frac{1}{\lambda t} \\) shrinks as the iteration count \\( t \\) climbs. The thing to hold onto is that the sign of each point's contribution is carried by \\( y_j \\) sitting right there in the sum, which means \\( \alpha_j \\) itself is meant to be an unsigned magnitude. The label tells you which side, and \\( \alpha \\) tells you how much. Keep that division of labor in mind, because the bug is precisely a violation of it.

When the margin on point \\( i \\) is violated, meaning \\( y_i f(x_i) < 1 \\), Pegasos applies the regularization shrink and a step toward fixing the violation, and when the margin is satisfied it applies the shrink alone. Written as updates, the two cases are

\\[ \alpha_i \leftarrow (1 - \eta \lambda)\,\alpha_i + \eta \qquad \text{(margin violated)} \\]

\\[ \alpha_i \leftarrow (1 - \eta \lambda)\,\alpha_i \qquad \text{(margin satisfied)} \\]

with the bias moving as \\( b \leftarrow b + \eta y_i \\) on a violation. The shrink term \\( (1 - \eta \lambda) \\) is the regularization, and it pulls every coefficient toward zero on every single step regardless of which branch you are in, which is what stops the coefficients running away.

The reference solution had two faults, and they compound. Here is the inner loop as it stood, with my change:

```python
decision = sum(alphas[j] * labels[j] * kernel_func(data[j], data[i]) for j in range(n_samples)) + b
if labels[i] * decision < 1:
-   alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
+   alphas[i] = (1 - (eta * lambda_val)) * alphas[i] + eta
    b += eta * labels[i]
+else:
+   alphas[i] = (1 - (eta * lambda_val)) * alphas[i]
```

Expand the original violation line and it reads \\( \alpha_i \leftarrow (1 - \eta \lambda)\,\alpha_i + \eta y_i \\), which differs from the correct update by that trailing \\( y_i \\). On its own a stray label looks harmless, but the decision function already multiplies \\( \alpha_j \\) by \\( y_j \\), so baking the sign into \\( \alpha \\) as well counts the label twice: a negative point pushes its coefficient negative, and then the \\( \alpha_j y_j \\) product flips it back, so the magnitude and the sign are now fighting each other through the same variable. The coefficients stop being unsigned magnitudes and start drifting toward \\( \pm 100 \\), which is exactly the runaway the regularization is supposed to prevent, and it is no longer preventing it because the update has quietly changed what \\( \alpha \\) means.

The second fault is the missing `else`. The original code only ever touched \\( \alpha_i \\) when the margin was violated, so a point that was comfortably classified never had its coefficient decayed at all, when Pegasos wants that \\( (1 - \eta \lambda) \\) shrink applied on every iteration. The fix is the two-line `else` that applies the shrink in the satisfied case, which is the half of the regularization that had simply gone missing.

What made this more than a one-line solution patch is that the hidden tests had been written against the broken solution, so they asserted the wrong answers, the saturated `[100.0, 0.0, -100.0, -100.0]` rather than the `[2.0, 2.0, 6.0, 1.0]` the stated algorithm actually produces. Fixing the solution without fixing the tests would just move the failure, so both had to change together, and the test outputs in the corrected version are the ones you get by running the algorithm the problem describes. The write-up is in [issue 592](https://github.com/Open-Deep-ML/DML-OpenProblem/issues/592) and the fix is up as [pull request 593](https://github.com/Open-Deep-ML/DML-OpenProblem/pull/593), which is in review rather than merged as I write this, so the specifics may yet shift before it lands.

## Takeaway

Practicing by solving a problem teaches you the algorithm. Practicing by repairing a broken one teaches you the algorithm and then asks you to prove it, because you cannot fix what you only half understand, and the cost of being wrong is a public pull request rather than a green checkmark only you ever see. Deep-ML is a good site for the first kind of practice and an unusually good one for the second, purely because the whole thing is open and the bugs are right there waiting for someone who reads carefully enough to notice them.
