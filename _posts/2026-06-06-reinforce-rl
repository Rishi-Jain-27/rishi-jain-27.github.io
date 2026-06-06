---
layout: post
title: "REINFORCE, landing on planets, and playing Flappy-Bird"
author: "Rishi Jain"
date: 2026-06-06
categories: projects
project_url: https://github.com/Rishi-Jain-27/lunar-lander-rl
project_label: Code
math: true
---

This is a quick blog post about the REINFORCE algorithm in reinforcement learning and the recent project I made with it. It also features mentions to another, previous project.

## REINFORCE: the top level.

REINFORCE stands for "REward Increment = Nonnegative Factor * Offset Reinforcement * Characteristic Eligibility". It is a foundational algorithm in policy gradient reinforcement learning.

To better understand it, it might be helpful to contrast it first with a baseline: Q-learning. We'll go through both Tabular- and Deep-Q learning.

Q-learning attempts to learn the value of specific state-action pairs. In other words, it just tries to maximize the reward gained from a particular state by choosing a particular action. In Q-learning, we learn which exact action to take in any state. Well, that's for tabular Q-learning, which just stores a table and brute-force calculates what action to take for what state. Deep Q-learning is a bit different, but relies on similar principles of learning what action to take in a given state. The goal is to maximize the total reward by picking the action that has the highest estimated "Q-value" for a given state.

For example, in my [Flappy-Bird DQN algorithm](https://github.com/Rishi-Jain-27/flappy-bird-dql) that I made by learning from [Johnny Code's tutorial](https://www.youtube.com/watch?v=arR7KzlYs4w), the DQN model learned when to and to not flap. We could, of course, do this mathematically by applying kinematics to the values of velocity and position provided by the environment (assuming that horizontal velocity and acceleration due to gravity is constant) and, after trial-and-error, eventually determine when to flap. For example, setting flap=True if position <= position of next pipe or if vertical velocity is greater than a certain value.

But that's arduous. Who would want to implement kinematics into Flappy Bird, of all games? No better game? And why waste so much compute on those calculations when we could have a neural network determine it?

The premise of Deep-Q Learning is using a Deep-Q Network, a fancy way of saying a neural network trained using loss values determined from the Q(uality) value, to have a neural network determine when and which action to take given a certain state of the environment. More often than not, this is more efficient and a better solution than deterministically accomplishing this task using kinematics and trial-and-error.

The way Deep-Q Learning goes about that is having two functions. One function, the policy function, learns. Note some key vocabulary: from now on, "policy" is a strategy that determines which action to take. It learns to pick the actions that maximize reward. What is reward? Well, reward is calculated from the other function — the target function. The target function helps calculate reward by outputting value, and it is simply a delayed copy of the policy function.

What Deep-Q learning does is it picks the action that maximizes the value output of the target function, thereby maximizing the reward. In the process, the target function copies the policy function. As a result, in Deep-Q Learning, the policy function learns to choose the action that will maximize the value from the target function in any given state.

(Note that these "functions" are neural networks. It's all just math.)

Now that we have the baseline of Q-learning defined, let's get into REINFORCE. Unlike Q-learning, which, if you recall, attempts to learn the value of state-action pairs, REINFORCE learns which action to take directly. REINFORCE learns the policy instead of tying it to a value function. 

## The Core Logic

In REINFORCE, the agent (the actor doing the actions) interacts with the environment (the setting where an AI "agent" operates) following a policy function. The agent collects a full list of all the states, actions taken, and rewards until a termination condition is reached. A termination condition is just what ends the agent's period of interaction with the environment. The agent's period of interaction with the environment is known as the episode.

So, in fancy terms, the agent collects a list of states, actions, and rewards until the episode terminates. These lists collected are sequential: the second element in each list would be the state two steps after the agent begins operating, the action taken, and the reward that action is thought to have provided.

The goal of REINFORCE is to update the parameters of the neural network that defines what policy the agent takes (also known as the policy network) to increase the probability of actions that led to a high amount of rewards cumulatively and lower the probability of actions that lead to low rewards.

Instead of calculating the immediate value like in Q-learning, REINFORCE tries to maximize actions that create high overall reward.

For example, if an action taken gave a resulting sequence of negative reward values, REINFORCE would minimize the probability that the policy network makes the agent take that action.

If an action taken gave a resulting sequence of positive reward values, especially high ones, REINFORCE would maximize the probability that the policy network makes the agent take that action.

## The Workflow

Here's what I just wrote in math.
1. Initialize policy parameters $\theta$ randomly.
2. Generate a trajectory (list of the lists of state, action, and reward):
Run the policy $\pi_\theta$ in the environment to collect a sequence: $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)$.
3. Calculate the return: For each time step $t$, calculate the total discounted return $G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k$.
4. Update the policy. Adjust the parameters $\theta$ using the Policy Gradient theorem. There's a lot of math for this, and that will be addressed in a future blog post.

## Key Characteristics

These are some distinguishing factors about REINFORCE.
1. The agent does not need to know the dynamics of the environment. It learns strictly through trial and error.
2. Monte Carlo based. Monte Carlo means this algorithm discovers optimal policies by using completed trajectories (in other words, by going through an entire episode). REINFORCE  waits until the end of the episode to calculate $G_t$ and perform the update. This makes it unbiased but often results in high variance, as the total reward $G_t$ can vary significantly between episodes.
3. On-policy. REINFORCE only updates the policy based on the data collected by the current version of the policy network. Once updated, the old data is discarded.

## Lunar Lander

Now, let's talk about something perhaps unrelated. Have you ever thought about landing a spaceship on the moon? Have you ever realized how difficult that might be with different gravity and rocky terrain?

If you have, then there's a pretty fun game for that on [Gymnasium](https://gymnasium.farama.org/), a website that gives access to very easy to create environments for Reinforcement Learning applications.

In the lunar lander game, a ship is landing and needs to ideally land between the two flags marking flat terrain on the ground. However, one wrong spike in acceleration with your thrusters (you have a left, right, and center thruster) could send you rotating into a crater, not an ideal outcome to say the least.

In this game, I implemented the REINFORCE algorithm. You can check out my code [here](https://github.com/Rishi-Jain-27/lunar-lander-rl).

I'd encourage you to atempt to train models using Reinforcement Learning in these environments. It was fun, and a bit of a challenge.

## Future Posts
Here are some upcoming blog posts:
- PPO technical explanation.
- Hindi-English Translator blogs. Features blogs about RoPE, beam-search, the transformer, and using Gradio.
- HuggingFace Small Models Hackathon blog post. Describes my process and what I learned.
