---
layout: post
title: "HackerRank Orchestrate 2026: a support triage agent that doesn't hallucinate"
author: "Rishi Jain"
date: 2026-05-03 12:00:00 -0500
categories: projects
project_url: https://github.com/Rishi-Jain-27/hackerrank-orchestrate-2026
project_label: Code
---

I entered the HackerRank Orchestrate 2026 hackathon and built a deliberately conservative, RAG-based support triage agent, which finished 298th out of 1,349 contestants. The agent reads an incoming support ticket, retrieves the relevant Markdown docs from the provided corpus, classifies the issue type and product area, and then decides whether to answer the ticket or escalate it to a human. The code is on [GitHub](https://github.com/Rishi-Jain-27/hackerrank-orchestrate-2026). The one design choice that shaped everything else was that I refused to let the language model make any of the routing decisions.

## Deterministic routing, model-generated prose

The agent comes in two halves, and the line drawn between them is the most important decision in the whole system. Everything safety-critical is handled by deterministic rules: the status of a ticket, whether it was answered or escalated or refused, the request type, the product area, and the escalation decision itself are all produced by rule-based logic that the model never touches, so the same input always yields the same routing output and the reason a ticket went where it did stays auditable after the fact. The model is left with exactly one job, which is to write the final response and its justification, grounded in the snippets the routing layer has already handed it. Models hallucinate, they get jailbroken, and given the chance they will confidently file a billing complaint as a documentation question, so pulling the routing out of the model entirely is the only way to know, once a ticket has gone wrong, why it went wrong.

## Retrieval that is company-filtered, path-aware, and override-aware

The retrieval layer does rather more than nearest-neighbour search over chunks. It is company-filtered, so a ticket from Company A only ever retrieves against documentation belonging to Company A and the agent never quotes one tenant's docs at another. It is path-aware, because the Markdown files live under structured paths and I pull product-area metadata out of those paths on ingest, on the reasoning that where a file sits in the tree carries real signal and throwing that away would be wasteful. And it respects nested overrides, so that when a deeper folder legitimately belongs to a different product area than its parent, the deeper override wins, which is what keeps retrieval from drifting wrong on the edge cases. By the time the model sees any snippets at all they have been scoped tightly to the right tenant and the right product area, before a single token is generated.

## The safety layers

On top of retrieval and rule-based routing I stacked a few guards, none of them large on its own. Ticket text is treated as untrusted data rather than as instructions, so the role-overrides and the "ignore previous instructions" and the embedded fake system prompts, the whole standard bag of tricks, get caught and neutralised before the model ever reads them. A risk-detection pass sends the known dangerous patterns, anything touching security, account-specific data, or credential-shaped strings, straight to escalation regardless of what the model might have preferred. A consistency check escalates whenever the rule-based classifier names one product area while the retrieved docs come predominantly from another, rather than papering over the conflict. And a postprocessing validator runs after the model has written its response, confirming that the output stays grounded in the retrieved snippets and consistent with the routing decisions, which is where hallucinated features and contradictions and quiet drift off the corpus get caught. Any one of those guards is small, and stacked together they are what stands between a demo and something I would be willing to put anywhere near a real support queue.

## What it answers and what it escalates

The behaviour follows the same conservatism as the architecture. The agent answers the simple, documented questions, the ones where the corpus plainly contains the answer and the routing is unambiguous, and it escalates anything account-specific, anything touching billing or security or an assessment, any outage, and anything vague or simply unsupported by the docs. That escalation list is the specification rather than a failure mode, because an agent that escalates the right twenty percent of tickets and answers the other eighty correctly is far more useful than one that tries to answer everything and gets a meaningful slice of it wrong. A confident wrong answer erodes the human team's trust in the agent and manufactures cleanup work that did not exist before it spoke.

## Takeaway

The interesting part of building an agent right now lives in deciding which decisions the model is allowed to make at all. I gave this one a single job, writing a response from context that had already been retrieved, already been routed, and already been validated, and then built everything around it to keep it inside that job.
