---
layout: post
title: "Double DQN: why the max leans high"
author: "Rishi Jain"
date: 2026-06-04
categories: notes
math: true
---

I had been watching a DQN agent learn a small control task and noticed that its predicted value for the early states kept climbing well past anything the actual returns could justify, the agent quietly convinced it was sitting on a goldmine of future reward that never arrived, and the gap between what it believed and what it collected was wide enough that the policy itself started to wobble. That optimism is baked into the shape of the Q-learning target rather than into any one line of code, and Double DQN is the small, almost embarrassingly cheap correction that takes most of it back out. These are the notes on why the standard target leans high and what the fix actually changes, with the mechanism drawn out rather than asserted. It assumes you are comfortable with the basics of Q-learning and DQN: the replay buffer, the separate target network whose weights lag the online network, and the Bellman update that ties a state's value to the immediate reward plus the discounted value of wherever you land next.

## The target DQN trains toward

A DQN keeps a network \\( Q(s, a; \theta) \\) that scores every action in a state, and it learns by nudging that score toward a target built from the next state. For a transition \\( (s_t, a_t, r_t, s_{t+1}) \\) the target is the reward you just received plus the discounted value of the best you can do from the next state, where "the best you can do" is read off the lagged target network \\( \theta^- \\):

\\[
y_t \;=\; r_t + \gamma \max_{a'} Q(s_{t+1}, a';\, \theta^-).
\\]

The loss is then the squared distance between this target and the network's current estimate of the action actually taken,

\\[
L(\theta) \;=\; \big( y_t - Q(s_t, a_t;\, \theta) \big)^2,
\\]

and the whole apparatus of replay buffers and target networks is machinery for making the regression on that single equation stable. The piece that matters here is the \\( \max_{a'} \\). At the next state we do not yet know which action we will take, so we assume we will take the best one and bootstrap from its value, which is exactly the optimality principle Q-learning is built on. The catch is that we are taking the max not of the true action values but of a network's noisy guesses at them, and the max of a set of noisy guesses is not a harmless operation.

## The max of noisy numbers leans high

Suppose, to strip the problem down to nothing, that we are at a state with \\( n \\) actions and every one of them is genuinely worth the same amount, which we will call zero. The true value of the best action is therefore zero, since they are all zero. Our network does not see those true values, it sees estimates corrupted by noise, so for each action it reports

\\[
Q(s, a) \;=\; 0 + \epsilon_a,
\\]

where the \\( \epsilon_a \\) are independent, zero-mean, and as likely to sit above the truth as below it. No single estimate is biased: averaged over many redraws each \\( Q(s, a) \\) comes out to zero, which is correct. The trouble starts the moment we apply the max. Taking the largest of the \\( n \\) estimates does not pick a random one, it deliberately seeks out whichever action got the most generous upward kick of noise on this particular draw, so the quantity we report is

\\[
\max_a Q(s, a) \;=\; \max_a \epsilon_a \;\ge\; 0,
\\]

and because the max of several zero-mean numbers is itself positive on average, we have

\\[
\mathbb{E}\Big[\max_a Q(s, a)\Big] \;>\; 0 \;=\; \max_a \mathbb{E}\big[Q(s, a)\big].
\\]

The expectation of the max sits strictly above the max of the expectations, which is just Jensen's inequality applied to the max, the max being a convex function. The estimator was unbiased action by action and the max turned it into an optimist, and notice that nothing about this needs the network to be badly trained: ordinary, well-behaved, zero-mean estimation noise is the entire cause, so the bias is there even on a network that is doing everything right.

It helps to put a number on it. Take the noise to be uniform on \\( [-c, c] \\), the simplest symmetric noise there is. The expected maximum of \\( n \\) independent draws from \\( [-c, c] \\) works out to a clean closed form,

\\[
\mathbb{E}\Big[\max_{a=1,\dots,n} \epsilon_a\Big] \;=\; c\,\frac{n-1}{n+1},
\\]

which you can read straight off:[^uniform] with one action there is no max to take and the bias is zero, with two it is \\( c/3 \\), and as \\( n \\) grows it climbs toward the full noise amplitude \\( c \\), because with enough actions one of them is nearly certain to have caught a kick close to the ceiling. For four actions and noise of amplitude one this is \\( 1 \cdot \tfrac{3}{5} = 0.6 \\), so an action genuinely worth nothing gets reported as worth 0.6 purely because we looked at four noisy guesses and kept the luckiest. And the effect refuses to stay contained to a single update: the inflated target feeds back into the next update as the thing the network is regressing toward, the network learns to reproduce the inflation, the inflated values feed the next max, and the optimism compounds through training rather than averaging out.

The widget below runs exactly this experiment. Every action is worth zero, you choose how many actions there are and how much noise corrupts the estimates, and each sample draws fresh noise. The top panel shows one draw, the bars being the noisy estimates and the tallest one being what the max selects. The bottom panel keeps the running average of that max over every sample you draw, and you can watch it pull away from the true value of zero and settle onto the dashed bias line at \\( c(n-1)/(n+1) \\). Add actions or crank the noise and the gap widens, which is the uncomfortable part: the bias is worst exactly where you have many actions and a network that has not yet learned them well, which is to say early in training on a large action space.

{% include viz/overestimation.html %}

The second trace on that bottom panel, the one sitting calmly on zero, is the fix, and it is worth understanding why it behaves so differently before we write down the formula it comes from.

## Decoupling selection from evaluation

The whole problem lives in the fact that one operation, the max, is quietly doing two jobs at once. It selects an action, the argmax, the one it thinks is best, and it evaluates that action, returning the value attached to it. When both jobs are done by the same noisy estimates the two become entangled in the worst possible way, because the action gets selected precisely on the strength of its noise and then has that same inflated noise read back as its value. The selection is biased toward whatever overestimated itself, and the evaluation then trusts that overestimate. To see the entanglement it helps to write the DQN max in a way that pulls the two jobs apart, since the max is nothing more than evaluating the argmax:

\\[
y_t^{\text{DQN}} \;=\; r_t + \gamma\, Q\Big(s_{t+1},\ \underbrace{\arg\max_{a'} Q(s_{t+1}, a';\, \theta^-)}\_{\text{selection}};\ \underbrace{\theta^-}\_{\text{evaluation}}\Big).
\\]

Written like this the issue is plain: the network that picks the action, \\( \theta^- \\), is the very same network that scores it, also \\( \theta^- \\), so any action it overrates is both more likely to be chosen and reported at its overrated height. Double DQN changes one thing and only one thing. It keeps the selection on the online network \\( \theta \\), the one you are actively training, and moves the evaluation to the target network \\( \theta^- \\):

\\[
y_t^{\text{DDQN}} \;=\; r_t + \gamma\, Q\Big(s_{t+1},\ \arg\max_{a'} Q(s_{t+1}, a';\, \theta);\ \theta^-\Big).
\\]

The online network proposes which action looks best at the next state, and the target network, which holds a different and older set of weights, is asked what that proposed action is actually worth. The reward, the discount, the bootstrap, the loss, the replay buffer, all of it is untouched, and the entire change is that the action index fed into the value lookup now comes from \\( \theta \\) instead of \\( \theta^- \\). The diagram below puts the two targets side by side.

{% include viz/ddqn-target.html %}

## Why a second opinion is harder to fool

The reason the swap helps is that overestimation now requires two networks to make the same mistake rather than one network to make it twice. In plain DQN, for an action to inflate the target it is enough that the single network overrates it, because that one overrating both wins the argmax and supplies the value. In Double DQN the action has to clear two separate hurdles: the online network has to overrate it enough to select it, and then the target network, holding different weights, has to independently also overrate it for the inflated value to come through. When the two networks' errors are not perfectly aligned, an action that fooled the online network into selecting it will, more often than not, be valued soberly by the target network, and the optimistic spike gets flattened on the way to the target.

Push the toy model through to see how complete the correction is in the clean case. Keep the state where every action is worth zero, but now give each action two independent noisy estimates, an online one \\( Q^A \\) and a target one \\( Q^B \\). The double estimator selects with \\( A \\) and evaluates with \\( B \\):

\\[
a^* \;=\; \arg\max_a Q^A(s, a), \qquad \text{value} \;=\; Q^B(s, a^*).
\\]

The index \\( a^* \\) is chosen entirely by \\( A \\)'s noise and knows nothing about \\( B \\)'s, so from \\( B \\)'s point of view it is just some action picked without reference to \\( B \\) at all, and the expected value \\( B \\) reports for it is therefore

\\[
\mathbb{E}\big[Q^B(s, a^*)\big] \;=\; 0,
\\]

dead on the truth, with the bias gone entirely. That is the flat trace in the widget: selecting with one estimator and evaluating with an independent one removes the systematic optimism completely, because the evaluation never gets to see, and so never gets to reward, the very noise that drove the selection.

The honest caveat is that DQN's two networks are not independent in this clean sense. The target network is a lagged copy of the online network, so their estimation errors are correlated rather than separate, and the more slowly the target trails the online weights the more their mistakes line up. Double DQN therefore shrinks the bias rather than erasing it the way the idealized double estimator does, and how much it shrinks depends on how decorrelated the lag manages to keep the two networks. In practice the lag is enough to take a large bias down to a small one, which turns out to be the difference that matters, and the [paper that introduced it](https://arxiv.org/abs/1509.06461) shows exactly this on Atari: value estimates that under DQN ran several times higher than the true returns come back down close to them under Double DQN, the games where DQN was most wildly optimistic are the ones where its score was worst, and fixing the estimate recovers a good deal of that lost performance with no extra network and no extra forward passes worth speaking of.

## What actually changes in the code

For all the discussion the implementation difference is two lines. In a standard DQN the next-state value is the max over the target network's own outputs:

```python
# DQN: the target network both selects and evaluates
with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values
    target = rewards + gamma * next_q * (1 - dones)
```

The `target_net(next_states)` call produces a value for every action, `.max(dim=1).values` collapses each row to its largest entry, and that single number is doing both jobs we pulled apart earlier. Double DQN splits the line in two, asking the online network for the action and the target network for its value:

```python
# Double DQN: online net selects the action, target net evaluates it
with torch.no_grad():
    next_actions = online_net(next_states).argmax(dim=1, keepdim=True)
    next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
    target = rewards + gamma * next_q * (1 - dones)
```

The first line asks the online network which action looks best at each next state and keeps the index rather than the value, the `keepdim=True` leaving it shaped as a column so it can be used to index into the next tensor. The second line is the only genuinely dense part: `target_net(next_states)` is again a value for every action, and `gather(1, next_actions)` reaches into each row and pulls out the one column named by `next_actions`, so instead of the row's maximum we get the target network's value for the action the online network chose, which is the whole of Double DQN in a single indexing operation. Everything downstream, the Bellman target, the loss, the optimizer step, is byte-for-byte identical to before.

## Takeaway

The overestimation is the max operator doing precisely what it was asked to do on inputs that happen to be noisy, an artifact of the operation rather than a flaw in the implementation, and Jensen's inequality guarantees that the result leans high however carefully you train. Double DQN does not argue with the max, it just refuses to let a single noisy estimate both nominate an action and grade it, handing the grading to a second network whose errors do not line up with the first. The clean double estimator removes the bias entirely and the lagged-target version that ships in practice removes most of it, and either way the cost is two lines of code, one tensor index, and not a single extra parameter, which is about as favorable as a correction ever comes.

[^uniform]: The clean form drops out of the order statistics. For \\( n \\) independent draws on \\( [0, 1] \\) the maximum has cumulative distribution \\( F(x) = x^n \\) and density \\( n x^{n-1} \\), so its mean is \\( \int_0^1 x \cdot n x^{n-1}\,dx = n/(n+1) \\). Rescaling that uniform onto \\( [-c, c] \\) by \\( \epsilon = 2c\,u - c \\) gives \\( \mathbb{E}[\max] = 2c\cdot\frac{n}{n+1} - c = c\,\frac{n-1}{n+1} \\).
