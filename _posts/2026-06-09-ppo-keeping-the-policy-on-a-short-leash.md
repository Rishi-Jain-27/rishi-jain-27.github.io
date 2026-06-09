---
layout: post
title: "PPO, and keeping the policy on a short leash"
author: "Rishi Jain"
date: 2026-06-09
categories: notes
math: true
---

In both the [REINFORCE post]({% post_url 2026-06-06-reinforce-rl %}) and the more recent [Soft Actor-Critic one]({% post_url 2026-06-08-soft-actor-critic-maximum-entropy %}) I left the same IOU lying around, a proper technical walk through PPO, and this is me finally paying it. Proximal Policy Optimization is the algorithm that a surprising amount of modern reinforcement learning quietly runs on, including the reinforcement-learning-from-human-feedback step that tunes a lot of large language models, and it earns that spot less through any one brilliant idea than through a sequence of careful, sensible repairs stacked on top of the plain policy gradient. The plan here is to start from that bare gradient and add the repairs one at a time, the value function, the advantage, the bias against variance compromise that becomes GAE, the data reuse that importance sampling buys, and finally the clip that is the part everyone actually means when they say PPO, so that by the end the whole objective reads as a series of fixes to obvious problems rather than a formula handed down from nowhere.

These notes lean on the REINFORCE post for the policy gradient and the actor-critic split, and they share a fair amount of furniture with the SAC post, the value function and the advantage and the entropy bonus among it, so it helps to have those somewhere in hand. I will rebuild the pieces I need as I go rather than assume all of them, but I will rebuild them quickly, since the interesting part is what PPO does once they are all on the table.

## The thing we are actually trying to do

An agent moves through an environment chasing some goal, and it does so in episodes, each episode chopped into discrete time steps. At every step the agent sits in some state, picks an action, and the environment answers with a new state and, sometimes, a reward, a single scalar saying how good that step turned out to be. Stack a whole episode of these together and you have a trajectory, which I will write as

\\[
\tau = (s\_0, a\_0, r\_1, s\_1, a\_1, r\_2, \dots, s\_T),
\\]

the full record of what the agent did and what the world handed back. The policy is what does the picking, and learning is the business of slowly reshaping it so the trajectories it produces collect more reward.

The awkwardness that makes reinforcement learning its own kind of hard is the thinness of the signal. There is no teacher labeling the right move at each step, only that scalar reward, often sparse, sometimes a single number arriving at the very end of the episode after a thousand choices have already been made. From that thin trickle of feedback we have to work out how to nudge the millions or billions of parameters inside the policy network so that next time it does a little better, and there is something faintly absurd about updating so many numbers on the strength of so few. Making that update steady, getting a stable, trustworthy push on an enormous parameter vector out of a noisy scalar, is more or less the entire problem, and PPO is one well-tested answer to it built on decades of prior work.

## A gradient with no target to aim at

The policy is a network \\( \pi\_\theta \\) that takes a state and returns a probability distribution over the available actions, and the agent acts by sampling from that distribution rather than reading off a single answer. If a teacher did exist, this would collapse into ordinary supervised learning: we would have a target distribution, all of its mass on the correct action and none anywhere else, we would measure the gap between that target and what the policy predicted with something like cross-entropy, and we would descend the gap. That is the comfortable world where you know the answer and only have to close the distance to it.

We do not live in that world here. We are never told the correct action, only handed a reward that grades the action we happened to take, so there is no target distribution to regress against and the supervised recipe has nothing to bite on. What we can do instead is weigh the log-probability of each action we actually took by a number that says how good that action turned out to be, and then step in the direction that raises the probability of the actions that did well and lowers the probability of the ones that did badly. Call that number the advantage \\( A\_t \\) for the moment, with the promise that I will pin it down properly in a few sections. The quantity we want to climb, at each step, is

\\[
\log \pi\_\theta(a\_t \mid s\_t)\, A\_t,
\\]

and the gradient of this, averaged over a batch of collected steps, is the policy gradient. The logarithm is there partly because differentiating it produces the \\( \nabla \pi / \pi \\) that the policy gradient theorem actually calls for, and partly because working in log-space is gentler on the arithmetic, but the shape to hold onto is plain enough: probability of the action, scaled by how good the action was. REINFORCE used the raw discounted return in the advantage's place, and most of what follows is the slow construction of a better thing to put there.

## Better than average, which means a value function

With no correct action to compare against, the most sensible move left is to judge the action we took against the average action available from that state. If our action beat the state's typical outcome, raise its probability, and if it came in below, lower it. That word average has to resolve to an actual number before it can do any work, and the number is the value of the state.

The value function gives, for a state, the return the agent can expect from there onward if it keeps following the policy, which is to say the discounted sum of all the future rewards averaged over every way the rest of the episode might unfold,

\\[
V^\pi(s\_t) = \mathbb{E}\_{\tau \sim \pi}\Big[\, \sum\_{k=0}^{\infty} \gamma^k\, r\_{t+k} \,\Big].
\\]

The \\( \gamma \\) is a discount factor between zero and one that makes rewards further in the future count for less, partly because a reward now is worth more than the same reward much later and partly because it keeps the sum from running off to infinity on a long episode. The discounted total of the rewards actually collected from step \\( t \\) onward in one concrete rollout, with no expectation taken, is the return \\( G\_t = \sum\_{k=0}^{\infty} \gamma^k r\_{t+k} \\), and the value function is just its expectation.

Once there are two networks in play they get the names that give the whole family its title. The policy is the actor, the thing that takes actions, and the value network is the critic, the thing that grades them, which is where actor-critic comes from. We train the critic the obvious way, by regressing its prediction onto the returns we actually saw, descending the mean squared error

\\[
L(\phi) = \big( V\_\phi(s\_t) - G\_t \big)^2
\\]

over the collected steps. The critic exists for one reason, to give us the baseline that turns a raw return into an advantage, so let me get to that.

## From a state's worth to an action's worth

The value function tells us what a state is worth on average, taking the rough with the smooth across every action the policy might pick from there. The gradient wants something narrower than that: the worth of the single action we genuinely took. That is the action-value \\( Q(s\_t, a\_t) \\), the expected discounted return given that from state \\( s\_t \\) we commit to action \\( a\_t \\) and only then carry on under the policy. The advantage is the difference between the two,

\\[
A(s\_t, a\_t) = Q(s\_t, a\_t) - V(s\_t),
\\]

which reads exactly as the name suggests: how much better the action we chose did than the state's average. A positive advantage means the action beat the baseline and its probability should go up, a negative one means it lagged and its probability should come down. Subtracting the value as a baseline has a quietly important side effect too, since it leaves the direction of the expected gradient untouched while noticeably shrinking its variance, which is the real reason the advantage is the right thing to weigh by rather than the bare return.

There is one catch, which is that we do not have \\( Q \\) lying around any more than we had the correct action. We have the critic's \\( V \\) and we have the rewards the agent collected, and out of those we have to manufacture an estimate of the advantage. How we do that turns out to be a genuine design choice with a real trade buried in it.

## Three ways to guess the advantage

The first option is to be patient and honest about it: wait for the whole episode to finish, add up the discounted rewards that actually arrived from step \\( t \\) onward, and use that real return in place of \\( Q \\), giving an advantage of \\( G\_t - V(s\_t) \\). This is the Monte Carlo estimate, and because it leans on rewards that genuinely happened it never systematically misstates the value, it is unbiased. The price is that an entire episode of random action choices and random rewards funnels into that one number, so it is noisy, high in variance, and it has the practical nuisances that you must wait for the episode to end before you can learn anything and that, when intermediate rewards are sparse and most of the \\( r\_{t+k} \\) are zero, the estimate carries very little signal for long stretches.

The second option is to stop waiting and let the critic finish the thought. Instead of summing the real future, take one real reward and then hand off to the value function for everything after, estimating \\( Q \\) as \\( r\_t + \gamma V(s\_{t+1}) \\). This is bootstrapping, using the critic's own estimate to stand in for the tail you have not seen, and it gives the advantage

\\[
\delta\_t = r\_t + \gamma V(s\_{t+1}) - V(s\_t),
\\]

the quantity called the temporal-difference residual. It is far steadier than Monte Carlo because all the noise of the distant future has been replaced by a single learned average, low variance, but it is biased, because the critic is wrong, especially early in training, and the estimate inherits whatever the critic currently believes. So the two extremes pull in opposite directions: Monte Carlo overcorrects on the noise of a single rollout, which is variance, while the one-step residual leans on a critic that has not learned yet, which is bias, and neither on its own is what you want.

Sitting between them is the whole family of \\( n \\)-step estimates, where you trust \\( n \\) real rewards and only then bootstrap,

\\[
A\_t^{(n)} = \sum\_{l=0}^{n-1} \gamma^l\, r\_{t+l} + \gamma^n V(s\_{t+n}) - V(s\_t).
\\]

At \\( n = 1 \\) this is the temporal-difference residual, low variance and biased, and as \\( n \\) grows toward the end of the episode it slides back into the Monte Carlo estimate, unbiased and noisy. The single number \\( n \\) is a dial between bias and variance, which is suggestive, but committing to one value of it is leaving information on the table, because the different \\( n \\) each see the trade slightly differently.

## GAE, a weighted blend of all three

The idea behind generalized advantage estimation is to refuse to pick a single \\( n \\) and instead take an exponentially weighted average over all of them, with a new parameter \\( \lambda \\) in \\( [0, 1] \\) controlling how the weight decays. Worked through, that average collapses into a strikingly clean form, just a discounted sum of the temporal-difference residuals we already have,

\\[
A\_t^{\text{GAE}} = \sum\_{l=0}^{\infty} (\gamma\lambda)^l\, \delta\_{t+l}, \qquad \delta\_t = r\_t + \gamma V(s\_{t+1}) - V(s\_t),
\\]

so in practice you compute the residual at every step and then accumulate them backward with a decay of \\( \gamma\lambda \\), which costs almost nothing.[^gae] The \\( \lambda \\) is the bias against variance dial in its final form, and the two ends of its range are exactly the two estimators we already met. Set \\( \lambda = 0 \\) and every term past the first dies, leaving \\( A\_t^{\text{GAE}} = \delta\_t \\), the one-step residual, low variance and high bias. Set \\( \lambda = 1 \\) and the weights become plain \\( \gamma^l \\), at which point the sum telescopes,

\\[
\sum\_{l=0}^{\infty} \gamma^l \delta\_{t+l}
= \sum\_{l=0}^{\infty} \gamma^l \big( r\_{t+l} + \gamma V(s\_{t+l+1}) - V(s\_{t+l}) \big)
= \sum\_{l=0}^{\infty} \gamma^l r\_{t+l} - V(s\_t)
= G\_t - V(s\_t),
\\]

because each \\( \gamma^{l+1} V(s\_{t+l+1}) \\) produced by one summand is canceled by the \\( -\gamma^{l+1} V(s\_{t+l+1}) \\) of the next, and the only value term left standing after the whole collapse is the leading \\( -V(s\_t) \\). That is precisely the Monte Carlo advantage, unbiased and high variance. So GAE is a continuous slider between the two estimators rather than a third thing, and lowering \\( \lambda \\) shifts weight toward the steadier, more biased end while raising it shifts weight toward the noisier, more honest end. Typical values sit close to but below one, around 0.95, keeping most of the low bias while trimming the worst of the variance.

## The training loop, in its wasteful first form

With the advantage settled we can write down a working algorithm. Start from initial parameters \\( \theta\_0 \\) for the policy and \\( \phi\_0 \\) for the value function, and then repeat a single loop: run the current policy in the environment to collect a batch of trajectories, compute the returns \\( G\_t \\) and the GAE advantages \\( A\_t \\) for every step in the batch, update the policy by ascending the policy gradient with those advantages, and update the value function by descending its mean squared error against the returns. Then throw the batch away and go back to the environment for fresh data.

That last step is the wasteful part, and it is worth dwelling on because fixing it is where PPO actually starts. The trajectories we just collected were expensive, every step of them a real interaction with the environment, and we used each one exactly once before discarding it. Supervised learning would never stand for this: there you sweep over the same dataset for many epochs, wringing every drop of signal out of each example. The reason we cannot simply do the same here is that the policy gradient is an on-policy quantity, valid only for data drawn from the policy that is currently being updated, so the instant we take one gradient step the policy shifts and the batch we collected belongs to a slightly different, now stale policy. To reuse it we need a way to keep using old data after the policy has moved on, honestly, without pretending the data came from the new policy when it did not.

## Reusing data without lying about it

The tool for that is importance sampling, which is the small identity that you can compute an expectation under one distribution using samples drawn from another, as long as you reweight by the ratio of the two,

\\[
\mathbb{E}\_{x \sim p}\big[ f(x) \big] = \mathbb{E}\_{x \sim q}\Big[ \frac{p(x)}{q(x)}\, f(x) \Big].
\\]

The reweighting corrects for the fact that the samples came from the wrong distribution, upweighting the regions \\( p \\) cares about more than \\( q \\) did and downweighting the rest. Applied here, the data was generated by the old policy \\( \pi\_{\theta\_{\text{old}}} \\) but we want the objective under the new policy \\( \pi\_\theta \\), so we reweight each step by the ratio of the two probabilities,

\\[
L(\theta) = \mathbb{E}\_{\text{old}}\Big[\, \frac{\pi\_\theta(a\_t \mid s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t \mid s\_t)}\, A\_t \,\Big].
\\]

It is worth naming that the logarithm from before has quietly gone, replaced by the bare probability ratio, which is what importance sampling actually asks for. Writing that ratio as

\\[
r\_t(\theta) = \frac{\pi\_\theta(a\_t \mid s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t \mid s\_t)},
\\]

the objective is just \\( \mathbb{E}[\, r\_t(\theta)\, A\_t \,] \\), and notice that at the very start of an update, when the new policy still equals the old one, the ratio is one everywhere and this reduces exactly to the plain policy gradient. As the new policy drifts away from the old during the update, the ratios move off one and the reweighting starts doing real work. This is not literally the original objective any more, it is a stand-in that agrees with it near the old policy and is built to be optimized with old data, which is why it is called the surrogate loss, a simplified and tractable version of the mathematically grounded thing it approximates. The payoff is that we can now run several gradient epochs over the same collected batch instead of one, which is most of the sample efficiency PPO has.

## The clip, which is the actual point of PPO

There is a catch hiding in that surrogate, and it is the catch the whole algorithm is named for. Importance sampling is only trustworthy while the two distributions stay close. Let the new policy wander far from the old one and the ratios swing wildly, the variance of the estimate explodes, and a single large advantage can drag the policy a long way in one update, off a cliff it cannot climb back from. The earlier and more theoretically careful answer to this was TRPO, which bolts a hard constraint onto the problem, requiring that the new policy stay within a trust region measured by the KL divergence from the old one, and enforces it with second-order derivative information, which works but is heavy machinery.

PPO's move is cruder and, in practice, better. It notices that the ratio \\( r\_t(\theta) \\) already is a measure of how far the policy has moved, one when nothing has changed and further from one the more the update has shifted the probability of that action, so rather than constrain the policy with an external KL term it simply clips the ratio inside the objective to the interval \\( [1 - \epsilon, 1 + \epsilon] \\) and takes the smaller of the clipped and unclipped versions,

\\[
L^{\text{CLIP}}(\theta) = \mathbb{E}\Big[\, \min\big(\, r\_t(\theta)\, A\_t,\ \ \text{clip}(r\_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon)\, A\_t \,\big) \Big].
\\]

The widget below is this objective for a single step, drawn against the ratio \\( r \\), with a dial for the sign of the advantage and the clip width \\( \epsilon \\). The dashed line is the unclipped \\( r \cdot A \\), a straight line through the origin that rewards moving the ratio as far as the advantage points. The gold line is the clipped objective, which follows the dashed line inside the shaded band and then goes flat.

{% include viz/ppo-clip.html %}

Take the two cases in turn. When the advantage is positive, the action was good and we want to raise its probability, which pushes the ratio above one, and the objective happily climbs with it until the ratio reaches \\( 1 + \epsilon \\), at which point the clip caps it and the line goes flat, so there is simply no gradient left to push the probability any higher and the update has no reason to keep moving in that direction. When the advantage is negative, the action was bad and we want to lower its probability, pushing the ratio below one, and again the objective rewards that only down to \\( 1 - \epsilon \\) before flattening off. The minimum is what makes this safe rather than merely capped: the clip removes the incentive to move further in the direction the advantage is pointing, but it deliberately does not flatten the other side, so if an update has accidentally pushed the ratio the wrong way the objective is still on its unclipped slope there, with a full gradient available to pull the mistake back. The clip only ever takes away the reward for overshooting, never the means to correct an undershoot, and that asymmetry is the whole trick. The leash is short, with \\( \epsilon \\) usually around 0.1 to 0.2, but it only tightens when the policy tries to bolt.

## A third term, for staying curious

The objective PPO actually optimizes carries two more terms alongside the clipped one. The value function still needs training, so its mean squared error rides along in the same loss, and there is an entropy bonus on top to keep the policy from collapsing onto a single action before it has explored enough to know that action is genuinely best,

\\[
L = \mathbb{E}\Big[\, L^{\text{CLIP}} - c\_1 \big( V\_\phi(s\_t) - G\_t \big)^2 + c\_2\, \mathcal{H}\big[ \pi\_\theta(\cdot \mid s\_t) \big] \,\Big],
\\]

with \\( c\_1 \\) and \\( c\_2 \\) small coefficients setting how much the value error and the entropy each matter. The entropy \\( \mathcal{H} \\) is the same expected surprise of the policy's own actions that ran through the SAC post, large when the distribution is spread and small when it has committed, and adding it to the objective gently penalizes premature confidence.

This is exactly the contrast I drew at the end of that post, now from the other side. In SAC the entropy is not a side term at all, it is written into the very definition of the value and the objective and its weight is tuned automatically toward a target, which is what makes SAC the more committed explorer. In PPO the entropy is a modest fixed nudge bolted onto the side of the clipped objective, enough to discourage early collapse but no more, which fits PPO's whole character of being the simpler, fewer-moving-parts option that you reach for when samples are cheap and you would rather not babysit a more elaborate machine.

## Takeaway

Read forward through the post and PPO is a stack of repairs, each answering a specific complaint about the one before it. The plain policy gradient has no target, so we weigh by an advantage. The advantage needs a baseline, so we add a value function and the actor-critic split. Estimating that advantage trades bias against variance, so GAE blends the whole family of estimators on a single \\( \lambda \\) dial. The on-policy gradient throws away expensive data, so importance sampling lets us reuse a batch across several epochs. And reuse lets the policy wander dangerously far in one update, so the clip keeps it on a short leash, removing the reward for overshooting while leaving every undershoot correctable. The entropy term is the small finishing touch that keeps it curious.

What PPO still is not, for all that machinery, is off-policy. The clip lets it squeeze a handful of epochs out of each batch, but the surrogate is only honest while the new policy stays near the old, so before long the data goes stale and has to be regenerated, which is the line that separates it from a buffer-hoarding method like SAC. That is the deliberate trade at the heart of it: a little reuse bought very safely, in exchange for the simplicity and reliability that make PPO the thing so much of the field, language-model tuning included, ends up running on. Next on the list is the Hindi-English translator series, with its own posts on RoPE, beam search and the transformer, and the write-up of the HuggingFace small-models hackathon.

[^gae]: The clean residual form is not obvious from the words alone, and the [GAE paper](https://arxiv.org/abs/1506.02438) is where the exponentially weighted average of the \\( n \\)-step estimates is shown to collapse into \\( \sum\_l (\gamma\lambda)^l \delta\_{t+l} \\). The telescoping at \\( \lambda = 1 \\) in the body is the cleanest special case of the same algebra.
