---
layout: post
title: "GRPO, and letting the group be the baseline"
author: "Rishi Jain"
date: 2026-06-09
categories: notes
math: true
---

The [PPO post]({% post_url 2026-06-09-ppo-keeping-the-policy-on-a-short-leash %}) built its objective up as a stack of repairs to the plain policy gradient, and the last and most expensive piece in that stack was the critic, a second network the size of a small policy whose only job was to supply a baseline. GRPO is what you get when you look hard at that critic in the specific setting of language model training, decide it is not earning its keep, and replace it with something almost embarrassingly simple. These notes are about that replacement: where GRPO sits in the pipeline that turns a raw pretrained model into something that reasons, what exactly it throws away, and why throwing it away is defensible rather than reckless.

I am going to lean on the PPO post fairly heavily for the reinforcement learning machinery, the policy gradient, the return, the baseline, the actor-critic split, and the clipped surrogate, so it helps to have that one somewhere in hand. I will rebuild the pieces I lean on as they come up, but quickly, since the interesting part is the one move GRPO makes once they are all on the table.

## Where this sits in the pipeline

It is worth placing GRPO in the larger sequence first, because the algorithm only makes sense once you know what kind of training it is doing and at what stage. A model starts life randomly initialised, a few billion parameters set to noise, knowing nothing. The first and longest stage is pretraining, where it learns to predict the next token across a very large slice of the internet, and out of that single objective it absorbs grammar, facts, and a great deal of latent capability, though nothing yet that makes it pleasant to talk to.

After pretraining the model will happily continue any text but will not reliably answer a question, so the next stage is instruction fine-tuning, supervised training on instruction and response pairs that teaches it the shape of being asked something and replying. This is still ordinary supervised learning, target tokens and cross-entropy, the comfortable world where the right answer is known.

To push past what supervised imitation can give, the field reaches for reinforcement learning, and this is where the two roads diverge. The first road is preference fine-tuning, where we train without explicit correct answers and use an implicit signal instead, usually human preference over pairs of responses. This is the [RLHF](https://arxiv.org/abs/2203.02155) recipe, reinforcement learning from human feedback, where a reward model is first trained to predict which response a human would prefer and then the policy is tuned against that learned reward. The core training algorithm underneath classic RLHF is PPO, and a more recent and lighter alternative that skips the separate reward model entirely is [DPO](https://arxiv.org/abs/2305.18290), direct preference optimization.

The second road is reasoning fine-tuning, and it is the one GRPO was built for. Unlike a creative task, where preference is the only handle you have, a maths problem or a unit-tested coding problem has one right answer and a deterministic way of checking whether the model found it. That changes the reward from a learned, fallible preference model into a verifier you can actually trust, which is the idea behind RLVR, reinforcement learning with verifiable rewards. DeepSeek brought RLVR to prominence with their open [R1](https://arxiv.org/abs/2501.12948) model, and the algorithm doing the weight updates there could in principle be PPO, but they used their own [GRPO](https://arxiv.org/abs/2402.03300) instead, which is a good deal cheaper to run. So GRPO lives at the reasoning end of the second road, updating the policy on a verifiable correctness signal, and everything below is about how it does that update.

## The gradient, and the baseline it needs

Recall the shape of the policy gradient from the PPO post, stripped to its essentials. The policy is a distribution over actions, here over the next token, and learning means reshaping it so that good actions become more probable and bad ones less so. After a response comes back and we score it, we want to push up the probability of the actions that led to a good score and push down the ones that led to a bad one, and GRPO and PPO are both members of the same policy gradient family that do this by following the gradient of the log-probability weighted by how good the action was.

The picture I find steadies the intuition is the parameter space itself. A real policy has billions of parameters, but pretend for a moment there are only two, \\( \theta\_1 \\) and \\( \theta\_2 \\), so that the expected reward of the policy is a surface sitting over that two-dimensional plane, and suppose it is some smooth hill. Initialising the policy drops us at one point on that surface, and a single training step is the act of measuring which way is uphill and taking a step in that direction. Do it enough times and you climb toward the peak, the parameters that make the policy collect the most reward.

Written out, one REINFORCE-style update reads

\\[
\theta \leftarrow \theta + \sum\_{t} \nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t)\, R(\tau),
\\]

summed over the time steps of the trajectory, with \\( R(\tau) \\) the reward of the final response. The sign of that reward sets the direction: a positive reward climbs the hill and raises the probability of the actions taken, a negative one walks back down and lowers them. This bare form only ever sees the single final reward, but most settings hand out reward along the way too, so the more careful quantity to weigh by is the return \\( G\_t \\), the sum of the rewards from step \\( t \\) onward,

\\[
G\_t = \sum\_{k=0}^{\infty} \gamma^k\, r\_{t+k},
\\]

with the discount factor \\( \gamma \\) in \\( [0,1] \\) shrinking the weight of rewards further in the future, partly because nearer reward is worth more and partly to keep the sum finite. Substitute \\( G\_t \\) for that flat \\( R(\tau) \\) and you have REINFORCE proper, the precursor that both PPO and GRPO are built on top of.

REINFORCE has one more ingredient that matters more than it first appears, which is that you subtract a baseline from the return before weighing by it. The baseline is what lets the update judge merit rather than absolute size: an action that scored 8 is only good if the typical action from that state scores less than 8, and without something to compare against, every positive reward looks like cause for celebration and the gradient is needlessly noisy. The PPO road answered the question of what that baseline should be with the actor-critic split, estimating it with a state-value function \\( V\_\phi \\), a second network with its own trainable weights that learns to predict the average return from each state. The policy is the actor because it acts and the value function is the critic because it grades, and the difference between the return and the critic's baseline is the advantage,

\\[
A\_t = G\_t - V\_\phi(s\_t),
\\]

the quantity that actually weighs the gradient in a modern method. This is the same advantage the whole second half of the PPO post was about estimating well.

The critic does its job, but it is not free. It roughly doubles the memory the training run needs, since you are now carrying a second large network and its optimiser state, it adds its own forward and backward passes to every step, and it adds a fair amount of cognitive overhead, a second thing to tune and a second thing that can quietly be wrong and poison the advantage. In a video game with dense per-step rewards the critic has a clear, learnable signal and earns all of that cost. The question GRPO asks is whether it earns it in language.

## The simplest baseline that could work

GRPO's answer is to go back to basics and pick a baseline so plain it almost looks like giving up. Two simplifications, taken together. First, instead of a per-step return \\( G\_t \\) it uses only the final reward \\( R \\) of the whole response, the same number for every token in the trajectory. Second, instead of a per-state baseline that the critic has to learn, it uses a single baseline shared across all time steps.

Both of these are concessions to the reality of the setting rather than carelessness. In a long-form answer to a maths problem there are no honest intermediate rewards: the verifier can tell you whether the final answer is right, but it cannot tell you that token 200 was worth more than token 199, and any per-step value you assign is something you invented. A video game showers you with genuine intermediate signal, a point here, a coin there, and a critic that learns to predict it is learning something real. Language mostly does not, so GRPO embraces the suspicion that our per-step value estimates are guesses dressed up as measurements and refuses to build on them.

That leaves the question of where the shared baseline comes from if not from a learned critic, and the answer is the move the algorithm is named for. Given an instruction, GRPO does not sample one trajectory, it samples a whole group of them, usually somewhere between 4 and 8 completions of the same prompt, and scores each one. The baseline is then simply the average reward across that group, the others serving as the control that the one in question is measured against. Rewriting the advantage with that baseline and a normalisation by the group's spread,

\\[
A\_i = \frac{R\_i - \operatorname{mean}(R\_1, \dots, R\_G)}{\operatorname{std}(R\_1, \dots, R\_G)},
\\]

where \\( R\_i \\) is the reward of the \\( i \\)-th completion and the mean and standard deviation run over the whole group of \\( G \\) of them. Every token in completion \\( i \\) gets this same advantage \\( A\_i \\). Because each response is judged relative to its sibling responses to the same prompt, the method is Group Relative Policy Optimization, and the entire critic network has been replaced by a mean and a standard deviation over a handful of samples.

It is worth grinding through a tiny example, because the mechanism is the kind of thing that hides behind the notation. Suppose for one prompt we sample four completions and the verifier scores them \\( R = (1, 1, 1, 0) \\), three correct and one wrong. The mean is \\( 0.75 \\), the variance is \\( \tfrac{1}{4}\big(3 \cdot 0.25^2 + 0.75^2\big) = 0.1875 \\), so the standard deviation is about \\( 0.433 \\). The three correct completions each get an advantage of \\( (1 - 0.75)/0.433 \approx +0.58 \\) and the single wrong one gets \\( (0 - 0.75)/0.433 \approx -1.73 \\). The lone failure is pushed down hard while the three successes are nudged up gently, which is exactly the right instinct: when most of the group already succeeds, getting it right is unremarkable and getting it wrong is the thing to correct, and the normalisation produces that asymmetry on its own without anyone hand-tuning it.

The widget below is one prompt's group laid out as bars, each bar a completion's reward, with the dashed line sitting at the group mean. The thing to notice is that the line is not yours to place: drag any bar and it slides to the new average, so every other completion's advantage moves at the same time, which is the entire content of the word relative. The number above each bar is its advantage, gold where the completion beat the group and red where it fell short.

{% include viz/grpo-baseline.html %}

Set it to "one wrong" to watch the \\( (1,1,1,0) \\) case above, where the single failure takes the hard negative push, and then to "all right", where the bars go level, the spread collapses, and every advantage is zero. That last case is worth sitting with: a prompt the model already solves every time, or one it always fails, gives the group no internal contrast, and a group with no contrast hands back no gradient. GRPO learns only from prompts where the completions actually disagree, which is a real and slightly awkward property of the method rather than a quirk of the widget.

## When DeepSeek did try intermediate rewards

The flat per-response reward is a choice, not a law, and the DeepSeek engineers did test the alternative, which is worth recording because it sharpens why the simple version won. They split a model's response into its separate reasoning steps and attached a reward model that scored each step on its own, so that reward arrived along the trajectory rather than only at the end, an approach called process supervision.

Under process supervision the advantage stops being constant across the whole response. At a token at time \\( t \\), you use the reward of whichever reasoning step that token belongs to, so a token in the second reasoning step is weighed by the second step's score, and the mean and standard deviation that normalise it now run over all the reasoning steps across the whole group rather than over whole-response rewards. It is the same group-relative idea applied at finer grain. The catch is that it reintroduces exactly the thing the simple version was trying to escape, a separate model passing judgement on intermediate states whose judgements you have to trust, and for verifiable reasoning the plain final-answer reward turned out to be the more robust place to stand.

## The loss, built on PPO's

With the advantage settled, the rest of the GRPO objective is mostly inherited. The naive policy gradient loss would be the sum over time steps of the log-probability of each action weighed by its advantage, but that loss has the variance problem the PPO post spent its second half on: an update can read too much into a noisy batch and lurch too far, and once the policy has lurched there is often no walking it back. The standard cure is to constrain how far a single update is allowed to move the policy, a trusted region around the current parameters, which TRPO first did with second-order derivative information and PPO then did far more cheaply by clipping.

GRPO takes PPO's clipped surrogate wholesale, so I will only recap it. Rather than the log-probability, PPO works with the ratio between the new policy's probability of an action and the old policy's, written

\\[
r\_t(\theta) = \frac{\pi\_\theta(a\_t \mid s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t \mid s\_t)},
\\]

which is one when the policy has not moved and drifts away from one as the update shifts that action's probability. The clipped objective takes the smaller of the unclipped ratio times the advantage and a version of the ratio pinned to the interval \\( [1-\epsilon, 1+\epsilon] \\),

\\[
L^{\text{CLIP}} = \min\\!\big(\, r\_t(\theta)\, A\_t,\ \ \operatorname{clip}(r\_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\, A\_t \,\big),
\\]

so that the objective stops rewarding the policy for moving once it has moved as far as the leash allows in the direction the advantage points, while leaving a full gradient available to undo a move in the wrong direction. The PPO post worked through both signs of the advantage carefully and the logic is identical here, so I will not repeat it. The one thing to keep in mind is that the advantage \\( A\_t \\) sitting inside this clip is now GRPO's group-relative one rather than a critic's.

GRPO then makes its one addition to PPO's loss, a term that compensates for what dropping the critic took away. It subtracts a KL divergence between the current policy and a fixed reference policy,

\\[
L^{\text{GRPO}} = L^{\text{CLIP}} - \beta\, D\_{\mathrm{KL}}\\!\big(\pi\_\theta \,\\|\, \pi\_{\text{ref}}\big),
\\]

with \\( \beta \\) a small coefficient setting how hard the term pulls. The reference policy \\( \pi\_{\text{ref}} \\) is the model as it stood before any of this reasoning fine-tuning began, the instruction-tuned starting point, frozen. The KL term measures how far the policy's distribution has drifted from that frozen original and penalises the drift, which keeps the model anchored near where it started even as the reward pulls it toward better reasoning.[^kl]

That anchor is doing real work, and it is the deliberate counterweight to the simplification at the heart of the method. The critic, for all its cost, was a stabilising influence, a learned sense of what each state was worth that kept the updates honest. Tear it out and replace it with a noisy group average and you have made the training cheaper but also twitchier, more prone to the policy running off somewhere strange in pursuit of reward. Tethering the policy to its pre-training-self with a KL penalty buys back a good deal of that lost stability for a fraction of the cost of a second network, which is the whole bargain GRPO is offering: drop the critic, keep a leash to the starting point, and accept a little more noise in exchange for roughly halving what the training run has to carry.

## Takeaway

GRPO is best read as a single, opinionated edit to PPO rather than a new algorithm. It keeps the clipped surrogate untouched, keeps the policy gradient skeleton untouched, and changes exactly one thing, the source of the baseline, swapping a learned value network for the average reward of a small group of sibling completions to the same prompt, then normalising by that group's spread to get the advantage. Everything else it does, the flat per-response reward and the KL tether to the frozen reference, follows from that one swap, the flat reward because there is nothing honest to put at the per-step level once the critic is gone, and the tether because something has to stand in for the stability the critic used to provide.

Whether the trade is worth it depends entirely on the setting, and the setting GRPO was made for is the one where it shines, verifiable reasoning, where the reward is a trustworthy verifier rather than a fallible preference model and per-step value estimates were always going to be fiction. That is the ground on which R1 was trained, and it is a fair part of why an open model could be trained on it at all. Still on the list ahead of these notes is the Hindi-English translator series, with its posts on RoPE, beam search and the transformer, and the write-up of the HuggingFace small-models hackathon.

[^kl]: GRPO does not estimate this KL with the textbook formula but with an unbiased low-variance estimator, \\( \frac{\pi\_{\text{ref}}(a\_t)}{\pi\_\theta(a\_t)} - \log \frac{\pi\_{\text{ref}}(a\_t)}{\pi\_\theta(a\_t)} - 1 \\), which is always positive and is cheaper to compute over sampled tokens than the full divergence. The [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) where GRPO was introduced has the details.
