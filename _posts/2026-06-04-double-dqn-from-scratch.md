---
layout: post
title: "Simply explained: double DQN"
author: "Rishi Jain"
date: 2026-06-04
categories: notes
math: true
---

I wrote up [Double DQN]({% post_url 2026-06-04-double-dqn-overestimation-bias %}) already, and reading it back I think I leaned on a fair amount of vocabulary you only have if you have already spent a while with Q-learning: replay buffers, target networks, bootstrapping, the Bellman update, all turning up in the first paragraph as though they were common nouns. This is the same idea told from much further back, with every term defined the moment it appears and the formulas taken apart one symbol at a time, because the actual insight behind Double DQN is small and rather pretty and does not deserve to be hidden behind a wall of jargon.

The only thing I will assume is that you can take an average of some numbers and that you are willing to follow a little arithmetic when it shows up. Everything else gets built here.

## What the agent is trying to do

Picture a little program, the agent, dropped into some world: a game, a maze, a car on a road. At each moment the world is in some configuration, and we call that configuration the state, written \\( s \\). The agent looks at the state and picks one move out of a fixed menu of moves, and a move is called an action, written \\( a \\).

Having taken the action, the agent receives a number called the reward, written \\( r \\), which is the world's immediate verdict on that move: a point scored, a step closer to the exit, a crash that costs it dearly. Then the world settles into a new state, the agent picks again, and the whole loop repeats.

The agent's actual goal is not to grab the biggest reward right now but to collect the most reward in total over everything still to come, and that total-from-here-on has a name, the return. There is one wrinkle worth stating early: a reward arriving many steps in the future is worth a little less to us than the same reward arriving now, partly because the far future is uncertain and partly because it keeps the sums finite, so we shrink each future reward by a constant factor for every step we have to wait.

That factor is the discount, written \\( \gamma \\) (gamma), and it is just a number between 0 and 1, commonly something like 0.9 or 0.99. A reward of 10 collected one step from now counts as \\( 0.9 \times 10 = 9 \\), the same reward two steps out counts as \\( 0.9 \times 0.9 \times 10 = 8.1 \\), and so on, each step multiplying in one more factor of \\( \gamma \\).

## A Q-value is one number: how good is this move, here

To choose well the agent needs some sense of how good each move is, and the object that carries exactly that is the Q-value. The Q-value \\( Q(s, a) \\) is a single number that answers one question: if I am in state \\( s \\) and I take action \\( a \\) right now, and from then on I keep playing as well as I can, what is the total discounted reward I should expect?

The Q in the name is just a conventional letter, originally for "quality," and you can read \\( Q(s, a) \\) out loud as "the quality of doing \\( a \\) in \\( s \\)." Higher is better, and if the agent knew the true Q-value of every move in every state it would never have to think hard again, because it could simply look at the menu of moves available in its current state and take whichever one carries the largest Q-value.

A small concrete picture helps. Suppose our agent is a car approaching a fork, and the three moves available are go left, go straight, go right. Imagine the true Q-values happen to be \\( Q(s, \text{left}) = 2 \\), \\( Q(s, \text{straight}) = 5 \\), \\( Q(s, \text{right}) = 1 \\). Reading those off, going straight is the best move from this state and the value of being in this state and acting well is 5, because acting well means taking the move worth 5. None of this requires the agent to understand why straight is best, only to have the three numbers, which is the whole appeal of learning Q-values in the first place.

The catch, of course, is that the agent does not start out knowing these numbers. It has to estimate them from experience, and the tool we hand it for that is a neural network, which for our purposes you can treat as a flexible function with a pile of adjustable internal numbers, its parameters, that eats a state and produces a guessed Q-value for every action.

We write those parameters as \\( \theta \\) (theta) and write the network's guess as \\( Q(s, a; \theta) \\), the semicolon just separating the inputs we care about, the state and action, from the parameters \\( \theta \\) that happen to define the function. Training the network means nudging \\( \theta \\) until those guessed Q-values come to match the true ones, and the entire subject of this post is one specific way that nudging goes subtly wrong.

## Learning a value by leaning on your own next guess

Here is the awkward part: we want the network to learn the true Q-values, but nobody ever hands us a true Q-value to compare against. All the agent ever directly observes is rewards, one step at a time, and the true Q-value is a statement about the entire discounted future, which it never gets to see laid out. The clever escape, and it really is the central trick of this whole family of methods, is to define what a Q-value should be in terms of the very next step plus the value of wherever you land, and then to lean on the network's own guess for that next-step value.

Think about what \\( Q(s, a) \\) ought to equal. You take action \\( a \\) in state \\( s \\), you collect the immediate reward \\( r \\), and you arrive in some next state which we will call \\( s' \\) (read "s prime," just a name for the state that comes after \\( s \\)). From \\( s' \\) onward, the best you can do is worth the largest Q-value available there. So the value of your original move should be the reward you just got plus the discounted value of playing well from the next state, which we can write as a target to aim the network at:

\\[
y \;=\; r + \gamma \, \max_{a'} Q(s', a').
\\]

Let me take that apart piece by piece, because every symbol earns its place. The \\( y \\) on the left is the target, the number we want the network's guess to match. The \\( r \\) is the reward we genuinely observed, a hard fact, no guessing involved. The \\( \gamma \\) is the discount from before, shrinking the future part.

Inside, \\( a' \\) (a prime) is a stand-in for "some action available in the next state," and \\( Q(s', a') \\) is the value of taking that action there. The operation \\( \max_{a'} \\) means: look at the Q-values of all the actions available in the next state \\( s' \\), and keep the largest one. So \\( \max_{a'} Q(s', a') \\) is "the value of the best move available next," which is exactly the value of arriving in \\( s' \\) and continuing to play well.

A number makes it solid. Say the agent takes a move, collects an immediate reward of \\( r = 1 \\), and lands in a next state where the three available moves are estimated to be worth 3, 5, and 2. The \\( \max \\) picks the 5, the discount is \\( \gamma = 0.9 \\), and the target is

\\[
y \;=\; 1 + 0.9 \times 5 \;=\; 1 + 4.5 \;=\; 5.5,
\\]

so we now have a concrete number, 5.5, to push the network's guess for the original move toward. Training is then nothing more than repeatedly computing such a target from a remembered experience, measuring how far the network's current guess sits from it, and adjusting \\( \theta \\) to close that gap a little. Do this across millions of remembered experiences and the guesses slowly settle toward the true Q-values. That is the entire engine, and the one part of it that will betray us is sitting right there in plain sight: the \\( \max \\).

## The largest of several honest guesses is not honest

Notice what the \\( \max \\) is being fed. It is not handed the true value of each next move, because we do not have those, it is handed the network's noisy guesses at them, and a guess from a half-trained network is the true value give or take some error.

Sometimes the guess lands a bit high, sometimes a bit low, and early in training that error can be large. Taking the maximum of a handful of such guesses turns out to be a quietly biased thing to do, and the cleanest way to feel why is to step entirely out of reinforcement learning for a moment.

Suppose a number, say 10, and suppose several friends each try to guess it. Each friend is honest, by which I mean precisely that their guess is 10 plus some wobble that is just as likely to land above 10 as below it, so if any one friend guessed over and over their guesses would average out to exactly 10, no bias at all.

Now you do the one thing that seems harmless and is not: instead of taking one friend's guess, or averaging the guesses, you keep only the single largest guess in the room. The largest guess does not belong to a random friend, it belongs to whichever friend happened to wobble upward the most this time, so by selecting it you are deliberately fishing out an over-guess. Across many rounds the largest guess averages to something above 10, even though every individual friend was averaging to exactly 10. Nobody lied, and yet the procedure leans high.

The widget below is exactly that experiment and nothing more. Every guesser is aiming at the true number 10, each guess is 10 plus a random wobble that is symmetric about zero, and we always keep the biggest guess in the round. The top panel shows one round with the kept guess marked, and the bottom panel tracks the running average of the biggest guess as you play more rounds, which you can watch lift off 10 and settle onto the dashed line. Add guessers or widen the wobble and the overshoot grows, since with more guesses in the room one of them is all the more likely to have caught a big upward kick.

{% include viz/luckiest-guess.html %}

We can pin the overshoot to an exact number when the wobble is simple. Take the wobble to be uniform on the range \\( [-c, c] \\), meaning every value between \\( -c \\) and \\( +c \\) is equally likely, where \\( c \\) is how wide the wobble can get. The average of the largest of \\( n \\) such guesses works out to a clean formula:

\\[
\text{average overshoot} \;=\; c \, \frac{n - 1}{n + 1}.
\\]

Read it off the way you would read a price. With \\( n = 1 \\) guesser there is no "largest" to fish out, you just have the one honest guess, and the formula gives \\( c \times 0 = 0 \\), no overshoot, exactly right. With \\( n = 2 \\) it is \\( c \times \tfrac{1}{3} \\).

With four guessers and a wobble of width \\( c = 1 \\) it is \\( 1 \times \tfrac{4 - 1}{4 + 1} = \tfrac{3}{5} = 0.6 \\), so a thing that is truly worth 10 gets reported as worth 10.6 purely because we looked at four wobbly guesses and kept the luckiest. As the number of guessers climbs, the fraction \\( \tfrac{n-1}{n+1} \\) creeps toward 1 and the overshoot approaches the full wobble width \\( c \\), because with enough guesses in the room one of them is nearly certain to land right up at the ceiling.

Now carry this straight back into the agent. In the target \\( y = r + \gamma \max_{a'} Q(s', a') \\), the actions in the next state play the role of the friends, the network's guessed Q-values play the role of their guesses, and the \\( \max \\) is you keeping the largest.

So even when the network is unbiased move by move, getting each individual Q-value right on average, the \\( \max \\) inside the target reports a value that leans high, and the more actions there are and the noisier the half-trained guesses are, the higher it leans. The proper name for this overshoot is overestimation bias.

It would be one thing if the overshoot stayed put, a fixed little error we could shrug off. It does not. The inflated target \\( y \\) is the number we just trained the network's guess to match, so the network learns to reproduce the inflation, and that inflated guess then feeds the \\( \max \\) in the next round of targets, which inflates those, which the network again learns to match. The optimism feeds on itself through training rather than washing out, and the agent can end up serenely confident that ordinary states are worth far more than the rewards it actually collects from them ever justify.

## One operation quietly doing two jobs

To fix the overshoot we first have to see precisely where it sneaks in, and it helps to notice that the \\( \max \\) is secretly doing two separate jobs in the same breath. The first job is choosing which next action to bank on: out of all the moves in the next state, which one looks best.

The second job is reading off how much that chosen move is worth, the value we actually plug into the target. The \\( \max \\) bundles these together, because "the largest Q-value" is at once a choice of action and the value attached to it.

There is a tidy way to write the two jobs apart. The operation that picks out which action is largest, as opposed to the largest value itself, is called the argmax, written \\( \arg\max \\). Where \\( \max_{a'} Q(s', a') \\) returns the largest value, \\( \arg\max_{a'} Q(s', a') \\) returns the action that achieves it, the winning move's name rather than its score.

If straight has the largest Q-value, the \\( \max \\) hands you straight's value and the \\( \arg\max \\) hands you the word "straight." Taking the max is therefore the same as first using the argmax to pick the winning action and then evaluating that action's value, which lets us rewrite the target with the two jobs visibly separated:

\\[
y \;=\; r + \gamma \, Q\big(s', \; \arg\max_{a'} Q(s', a')\big).
\\]

Look at what this says. The inner \\( \arg\max_{a'} Q(s', a') \\) selects the action that the guessed Q-values think is best, and the outer \\( Q(s', \cdot) \\) evaluates that selected action by reading off its guessed value.

And here is the trap laid bare: it is the same set of guesses doing both. The action gets chosen because its guess is large, and then that same large guess is read back as its value, so an action that got an upward kick of noise is selected precisely on the strength of that kick and then has the kick reported as its worth. Selection is rigged toward whatever overshot, and evaluation then trusts the overshoot. That double use of one noisy guess is the whole disease.

## A second opinion is harder to fool

Once you see the disease the cure almost writes itself: stop letting one set of guesses both pick the action and grade it. Have one estimator choose the action and a second, independent estimator say what that chosen action is worth.

The action still gets picked on the strength of the first estimator's noise, but the value now comes from the second estimator, which had no part in the selection and so has no reason to be sitting high on that particular action. This is the double estimator idea, and it is the entirety of Double DQN.

Let me show, with the cleanest possible toy, how completely the second opinion defuses the bias. Take a state where every move is genuinely worth the same amount, zero, so the honest answer for "value of the best move" is also zero. Give each move two independent guesses: an \\( A \\)-guess and a \\( B \\)-guess, both honest and both noisy, drawn separately so that knowing one tells you nothing about the other.

We select the move using only the \\( A \\)-guesses, taking the action whose \\( A \\)-guess is largest, and then we report its value using only the \\( B \\)-guess of that same chosen move. The selected move is the one that happened to make \\( A \\) wobble up the most, but the \\( B \\)-guess we read for it knows nothing about \\( A \\)'s wobble, so from \\( B \\)'s point of view we have simply pointed at some move and asked for its honest guess, and an honest guess of a move truly worth zero averages to zero.

The overshoot is gone. Selecting with one estimator and evaluating with an independent one removes the lean entirely, because the estimator doing the grading never gets to see, and so never gets to reward, the very wobble that won the selection.

That second, calm trace is the one already sitting on zero in the widget below, which runs the reinforcement-learning version of the same experiment. Every action is truly worth zero, you set how many actions there are and how much noise corrupts the guesses, and each sample draws fresh noise. The bars are the first estimator's guesses, the dot is the independent second estimator's guess for each action, the top panel marks the action the first estimator selects, and the bottom panel tracks two running averages over many samples: the single-estimator value, which is the plain agent taking the max and which climbs to the overshoot line, and the double-estimator value, which selects with the first and evaluates with the second and which sits flat on zero where it belongs.

{% include viz/overestimation.html %}

In a real agent we do not keep two fully separate networks just for this, because that would be a lot of extra machinery. We already have two networks lying around for an unrelated reason: training is steadier if the targets are computed from a slightly stale copy of the network rather than the one we are actively changing, so most agents keep the live network being trained, called the online network with parameters \\( \theta \\), alongside a lagging copy whose parameters, written \\( \theta^- \\), are refreshed from the online ones only every so often.

The lag exists to stabilise training, but it conveniently hands us a second, not-quite-identical set of guesses, and Double DQN simply puts the two existing networks to work on the two jobs. The online network selects the action and the lagging copy evaluates it:

\\[
y^{\text{Double}} \;=\; r + \gamma \, Q\big(s', \; \arg\max_{a'} Q(s', a'; \theta); \; \theta^-\big).
\\]

The change from the plain target is small enough to point at: the action fed into the value lookup is now chosen by \\( \theta \\), the online network, while the value itself is read from \\( \theta^- \\), the lagging copy. The reward, the discount, the whole rest of the procedure is untouched. The diagram below sets the plain target and the Double DQN target side by side so you can see that the skeleton is identical and only the who-does-what at the next state has moved.

{% include viz/ddqn-target.html %}

## The honest caveat, and the cost

I have to be straight about one thing, because the clean toy above was a little too kind. In that toy the two estimators were fully independent, which is what made the overshoot vanish completely.

The online network and its lagging copy are not independent in that clean way: the copy is literally an older version of the same network, so when the online network is sitting high on some action, the slightly-older copy was very probably sitting high on it too, their errors leaning the same direction rather than canceling. Because of that overlap, Double DQN shrinks the overshoot rather than erasing it the way the idealised pair of independent estimators would, and how much it shrinks depends on how different the lag manages to keep the two networks.

In practice the lag is enough to turn a large bias into a small one, which turns out to be the difference that matters, and the [paper that introduced it](https://arxiv.org/abs/1509.06461) shows exactly this on the Atari games: value estimates that under the plain agent ran several times higher than the returns it actually collected come back down close to reality under Double DQN, and the games where the plain agent was wildest with its optimism are the ones where fixing the estimate buys back the most performance.

What does all this cost? Almost nothing, which is the part I find genuinely lovely. Both networks already exist. In the plain agent the next-state value is the largest entry of the lagging copy's own guesses, the max and the value coming from one network:

```python
# Plain DQN: the lagging copy both picks the next action and grades it
with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values
    target = rewards + gamma * next_q * (1 - dones)
```

Double DQN splits that into two short lines, asking the online network which action looks best and the lagging copy what that action is worth:

```python
# Double DQN: the online net picks the action, the lagging copy grades it
with torch.no_grad():
    next_actions = online_net(next_states).argmax(dim=1, keepdim=True)
    next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
    target = rewards + gamma * next_q * (1 - dones)
```

The first line asks the online network for the best next action and keeps the action's index rather than its value. The second line is the only dense bit: the lagging copy produces a guessed value for every action, and `gather` reaches into each row and pulls out the one value belonging to the action the online network chose, so instead of each row's largest value we get the lagging copy's value for the online network's pick.

Everything after that, the discount, the target, the gap the network is trained to close, is identical to before. No extra network, no extra forward pass worth mentioning, two lines and a single indexing operation.

## Takeaway

The overshoot was never a bug in anyone's code, it was the \\( \max \\) doing exactly what it was told on inputs that happened to be noisy, and the largest of several honest-but-noisy guesses leans high as surely as the largest of several friends' guesses at the number 10 does. Double DQN does not argue with the \\( \max \\), it just refuses to let a single noisy guess both nominate an action and grade it, handing the grading to a second network whose errors do not line up with the first, and the result is a much soberer value estimate for the price of two lines of code and not one extra parameter.

This was the gentle pass, deliberately slow and light on notation. If you want the same story told in the denser, more formal register, with the expectation-of-the-max inequality and the correlated-error analysis spelled out properly, it is in the [original Double DQN notes]({% post_url 2026-06-04-double-dqn-overestimation-bias %}).
