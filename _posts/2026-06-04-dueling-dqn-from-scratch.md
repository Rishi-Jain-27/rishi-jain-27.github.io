---
layout: post
title: "Simply explained: dueling DQN"
author: "Rishi Jain"
date: 2026-06-04
categories: notes
math: true
---

This is the gentle companion to the [Dueling DQN notes]({% post_url 2026-06-04-dueling-dqn-value-and-advantage %}), the same architecture explained from much further back, every term defined as it turns up and every formula taken apart symbol by symbol. The denser version assumed you were already at home with how a DQN agent works, so here I will rebuild just enough of that footing to stand on, and the only thing you really need is a willingness to follow a small worked example when one shows up.

If you have read the [companion piece on Double DQN]({% post_url 2026-06-04-double-dqn-from-scratch %}) the opening will feel familiar, since both rest on the same handful of ideas, and you can skim the next two paragraphs.

## The one number per move, and why it is sometimes wasteful

An agent is a program acting inside some world: a game, a maze, a car on a road. At each moment the world is in some configuration called the state, written \\( s \\), and the agent picks one move from a fixed menu, a move being called an action, written \\( a \\).

The thing the agent learns is a Q-value, written \\( Q(s, a) \\), which is a single number saying how good it is to take action \\( a \\) in state \\( s \\) and then carry on playing well, measured as the total reward it can expect to gather from here onward. Higher means better, and a fully trained agent simply reads the Q-values of the moves available in its current state and takes the largest.

So a plain agent keeps one function \\( Q(s, a) \\) that, fed a state, returns a separate number for every action, and the only feedback it ever gets is how wrong those numbers were. That is a fair thing to ask for when the move genuinely matters, but a great many states are not like that.

Picture the car on a long empty stretch of road, no other traffic and no junction in sight: nudging left, nudging right, holding straight, none of it changes how the drive goes in any meaningful way, and the single fact worth knowing about this state is that it is a comfortable, safe, moderately good place to be.

A plain agent cannot just say that. To express "this state is worth about 8, and every move keeps it at about 8" it has to separately pin down \\( Q(s, \text{left}) \approx 8 \\), \\( Q(s, \text{straight}) \approx 8 \\), \\( Q(s, \text{right}) \approx 8 \\), learning each of those numbers from its own slow trickle of experience and keeping them agreeing with one another by luck rather than by design. It is doing three jobs to say one thing.

## Two questions hiding inside one Q-value

The thought behind the dueling architecture is that \\( Q(s, a) \\) is really answering two quite different questions at once, and the two have different shapes. The first question is how good it is to be in this state at all, regardless of which move you make, which is a single number attached to the state and the same for every action.

The second is how much better or worse each particular move is than a typical move from here, which is where all the dependence on the action actually lives. Give the first a name, the value of the state, written \\( V(s) \\), and the second a name, the advantage of the action, written \\( A(s, a) \\), and the claim is simply that the Q-value is the two added together:

\\[
Q(s, a) \;=\; V(s) + A(s, a).
\\]

Each symbol is worth saying plainly. \\( V(s) \\) is the shared part, one number for the whole state, carrying "how good is it to be here." \\( A(s, a) \\) is the per-move part, one number for each action, carrying "how much does choosing this move shift things up or down from the state's baseline."

The word advantage is meant literally: a positive \\( A(s, a) \\) means this move is better than the state's baseline and a negative one means it is worse, so the advantage measures a move's edge over the typical move rather than its absolute worth. On the empty road the value learns the single number 8 and every advantage sits near zero, which is a far more honest description than three separately wobbling action values that happen to land on the same spot.

A small numerical example makes the split tangible. Suppose a state with three moves whose Q-values are \\( Q = (2, 2, 5) \\), two moves equally mediocre and a third clearly better. The baseline of this state, the average of its moves, is \\( (2 + 2 + 5)/3 = 3 \\), so we could set \\( V(s) = 3 \\) and then the advantages, each move's Q-value minus that baseline, come out as \\( A = (2 - 3, \, 2 - 3, \, 5 - 3) = (-1, -1, 2) \\).

The two mediocre moves sit one below the baseline and the good move sits two above it, and adding the value back to each advantage returns the originals: \\( 3 + (-1) = 2 \\), \\( 3 + (-1) = 2 \\), \\( 3 + 2 = 5 \\). The pair \\( V = 3 \\) and \\( A = (-1, -1, 2) \\) carries exactly the same information as the three Q-values, only split into the part the whole state shares and the part that distinguishes the moves.

## Building the split into the network

The architecture takes that split literally and bakes it into the shape of the network. Before saying how, a word on the network itself.

A neural network here is a stack of layers that transforms its input step by step, and the early layers that turn the raw observation, the pixels of the screen or the readings of the sensors, into a compact set of useful numbers are called the feature extractor or the trunk, and they are doing the unglamorous work of digesting the state into something the later layers can use. A plain agent runs the trunk and then feeds its output into a single final layer, the head, that emits one Q-value per action.

The dueling network keeps the trunk exactly as it was and forks it into two heads after it, each head being a small stack of its own layers. We will call each fork a stream. One, the value stream, squeezes the trunk's features down to a single number, the value \\( V(s) \\). The other, the advantage stream, produces a whole vector of numbers, one advantage \\( A(s, a) \\) for each action.

A final step called the aggregation recombines the single value and the vector of advantages back into one Q-value per action, and from the outside the network still takes a state and emits a Q-value per action just as before, so it drops into an existing training setup without anything else having to change. The picture below lays out the trunk, the two streams, and the aggregation.

{% include viz/dueling-streams.html %}

To keep the bookkeeping straight in a moment, give each part its own parameters, the adjustable internal numbers that training tunes. Let the shared trunk have parameters \\( \theta \\), the value stream its own \\( \beta \\) (beta), and the advantage stream its own \\( \alpha \\) (alpha).

So the value head computes \\( V(s; \theta, \beta) \\) and the advantage head computes \\( A(s, a; \theta, \alpha) \\), the two of them sharing the trunk \\( \theta \\) and parting ways only in the small heads on top. None of this is expensive: a couple of extra layers and one scalar output is the whole added cost, and the work of running the network forward and adjusting it backward is the same order as before.

## One thing observed, two things to pin down

The naive split, the bare \\( Q = V + A \\) we wrote above, has a quiet problem that surfaces the moment you ask what the training process can actually see. Training works by comparing the network's Q-value for a move against a target value built from the reward and the next state, and turning the gap between them into a single number measuring how wrong the network was, called the loss, which it then tries to make small by adjusting the parameters.

The point worth holding onto is that the loss is built only from \\( Q \\), the recombined output, and never from \\( V \\) and \\( A \\) on their own. It can tell when the sum \\( V + A \\) is wrong, but it has no way to tell whether \\( V \\) and \\( A \\) individually are the numbers we intended.

And that blind spot is fatal to the clean interpretation, because endlessly many different splits give the very same \\( Q \\). Take any constant number \\( c \\) at all, add it to the value and subtract it from every advantage, and the Q-values come out completely unchanged,

\\[
Q(s, a) \;=\; \big( V(s) + c \big) + \big( A(s, a) - c \big),
\\]

since the two \\( c \\) terms cancel the instant you add the parts back together. The network is therefore free to park \\( V(s) \\) at any height it pleases and quietly soak up the difference in the advantages, and because the loss is blind to which split was chosen, nothing pushes the training toward the one we wanted, the one where \\( V \\) genuinely means the value of the state.

The value stream could drift off to some meaningless offset while the advantage stream silently compensates, and the whole reason we built the two streams, to read \\( V \\) as the state's worth, would quietly evaporate with the loss reporting no complaint at all. In the jargon this is called an identifiability problem: the thing we want, the split, is not pinned down by the thing we can measure, the sum.

The widget below is that ambiguity made draggable. A single state has four fixed Q-values, the four bars, and those bars are the only thing the loss ever sees, so they never move. The waterline is the value \\( V \\), and each move's advantage is the gap from its bar's top down to that line, drawn gold where the bar pokes above the line and red where it sits below.

Drag the waterline anywhere you like: \\( V \\) changes, all four advantages change with it, and yet every recombination \\( V + A \\) lands back on exactly the same bar, which is the whole problem captured in one motion. The entire vertical range gives an equally good split, and nothing in the bars themselves tells you where the waterline ought to sit.

{% include viz/dueling-decomp.html %}

## Pinning the waterline with one rule

The cure is to stop leaving the split free and impose a single rule that selects exactly one waterline out of the infinite family. Dueling does this by subtracting the average advantage inside the aggregation, so that what the network actually emits is

\\[
Q(s, a) \;=\; V(s) + \left( A(s, a) - \frac{1}{\vert \mathcal{A} \vert} \sum_{a'} A(s, a') \right).
\\]

Let me unpack the new piece, since it looks heavier than it is. The symbol \\( \vert \mathcal{A} \vert \\) is just the number of actions available, so if there are four moves then \\( \vert \mathcal{A} \vert = 4 \\). The expression \\( \sum_{a'} A(s, a') \\) means add up the advantages over all the actions, the \\( \sum \\) being a summation sign and \\( a' \\) running across every move.

Divide that total by the number of actions and you have \\( \frac{1}{\vert \mathcal{A} \vert} \sum_{a'} A(s, a') \\), which is nothing more exotic than the average advantage. So the bracketed term is each raw advantage with the average advantage subtracted off, which is a standard move: subtracting the mean from a set of numbers forces the leftover numbers to average to zero.

That is the entire trick. No matter what the advantage stream emits, the advantages that actually go into the Q-value are forced to sum to zero, and that single constraint nails the waterline to one specific height.

Which height? Average both sides of the equation over the actions. On the right the bracketed term averages to zero by construction, so the average of \\( Q(s, a) \\) over the moves equals \\( V(s) \\) on its own:

\\[
\frac{1}{\vert \mathcal{A} \vert} \sum_{a} Q(s, a) \;=\; V(s).
\\]

So the value stream is pinned to mean exactly the average Q-value of the state, which is a thoroughly sensible thing for "the value of the state" to mean. In the waterline widget this is the button labelled "V to mean(Q)": it drops the line straight onto the average of the four bars, the one height where the gold gaps above and the red gaps below balance out and the advantages sum to zero.

Run it back through the worked example to watch the pinning bite. The state with Q-values \\( (2, 2, 5) \\) has average \\( 3 \\), so the mean-subtracted split is \\( V = 3 \\) with advantages \\( (-1, -1, 2) \\), which do sum to zero, and recombining gives \\( (2, 2, 5) \\) back.

Now suppose the value stream had drifted off to \\( V = 10 \\) instead: to still reconstruct the same Q-values it would need advantages \\( (2 - 10, \, 2 - 10, \, 5 - 10) = (-8, -8, -5) \\), and these recombine to the identical \\( (2, 2, 5) \\). Both splits are perfect as far as the loss can tell, which is the disease, but their advantages sum very differently: the first sums to zero, the second sums to \\( -8 - 8 - 5 = -21 \\). Impose the rule that the advantages must sum to zero and the second split is simply unreachable, so the value stream has nowhere left to drift and is forced to settle on the honest 3.

## Why the average and not the largest

The average is not the only rule that would pin the waterline, and there is a rival that looks more principled at first glance: subtract the largest advantage rather than the average,

\\[
Q(s, a) \;=\; V(s) + \left( A(s, a) - \max_{a'} A(s, a') \right),
\\]

where \\( \max_{a'} A(s, a') \\) means the biggest of the advantages. This rule forces the advantage of the best move to be exactly zero, which makes \\( V(s) \\) equal to the largest Q-value, the value of playing optimally, and that matches the textbook definition of advantage in which the best action has no advantage over itself.

In the waterline widget it is the "V to max(Q)" button, dropping the line onto the tallest bar so every advantage is zero or negative. Cleaner meaning, on paper.

The reason the average wins in practice is stability, and it comes from what each rule does to the value stream's target as the network learns. With the largest-advantage rule, the waterline is tethered to whichever move currently looks best, so the instant the advantage stream changes its mind about the top move, even by a hair, the waterline leaps to a different bar and \\( V(s) \\) lurches with it.

Early in training the best move keeps flipping as the noisy advantages reshuffle, and the value stream ends up chasing a target that keeps teleporting. The average rule ties the waterline to the average instead, which slides smoothly as the advantages adjust and does not care which move is momentarily on top, so the value stream sees a calm target and settles without the thrashing.

The widget below shows exactly that contrast. The four thin lines are the four moves' Q-values, drifting as the network trains and crossing one another as it changes its mind about which move is best.

The gold line is the average of the four, the value the network learns under the rule dueling actually uses, and it glides. The red line is the largest of the four, the value it would learn under the largest-advantage rule, and it snaps to a fresh corner every single time the best move changes, dragging the value stream's target along with it. Play it and watch the red line kink while the gold line stays smooth, then switch to "actions barely matter" and the curves bunch up, both lines flatten, and the gap between the two rules nearly closes.

{% include viz/duel-stability.html %}

The [original dueling paper](https://arxiv.org/abs/1511.06581) tries both rules and reports the average as the steadier of the two, and the only price is that \\( V(s) \\) now means the average Q-value of the state rather than the best one. That is a perfectly usable meaning, and it costs nothing in the end, because the constant offset between the two interpretations washes straight out when the parts are recombined into Q-values anyway.

## What the split actually buys

With the split well posed, the payoff is in how the learning gets spread around. In a plain agent, each visit to a state delivers feedback about the one move that was actually taken, so each Q-value is refined slowly, one visit at a time, and in a state with many moves most of them sit waiting their turn.

The dueling value stream is different: because the average-subtracting aggregation routes a piece of every move's feedback back into the single shared \\( V(s) \\), the value stream is touched on every visit to the state no matter which move was taken, so it gathers evidence about the state's worth far faster than any single Q-value could.

In the many states where the worth of being there is most of the story and the choice of move barely matters, that is most of the learning done early and cheaply, the value stream learning the shared number while the advantage stream learns a vector close to zero, and the agent never has to laboriously discover one move at a time that the moves were interchangeable all along.

Because the change is purely a matter of the network's shape and says nothing about how targets are computed or how past experience is replayed, it stacks cleanly with the other DQN improvements rather than competing with them. It composes directly with [Double DQN]({% post_url 2026-06-04-double-dqn-from-scratch %}), which leaves the architecture untouched and instead fixes the optimism in the target by having one network pick the next move and another grade it, and the two together are usually just called a dueling double DQN.

The two are patching genuinely different leaks, one the shape of the function and the other the bias of the target, so adding them up is free and standard, which is why the dueling architecture is rarely run on its own.

## Takeaway

The dueling architecture is a bet that the worth of a state and the relative merit of each move are different enough in shape to deserve different parts of the network, a single shared number for the first and a per-move vector for the second. Written naively as \\( V + A \\) the bet does not pay, because the loss only ever constrains their sum and the split is free to drift, so the real content of the idea is the average subtraction that pins the value stream to the average Q-value and forces the advantages to sum to zero.

That one rule is what turns a redundant pair of streams into a faster learner, and it costs a couple of extra layers and not a single change to the target the network trains toward. The denser treatment, with the parameter bookkeeping and the stability argument written out more formally, is in the [original dueling notes]({% post_url 2026-06-04-dueling-dqn-value-and-advantage %}).
