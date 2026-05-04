---
layout: post
title: "HackerRank Orchestrate 2026: Building a Support Triage Agent That Doesn't Hallucinate"
author: "Rishi Jain"
date: 2026-05-03 12:00:00 -0500
categories: projects
project_url: https://github.com/Rishi-Jain-27/hackerrank-orchestrate-2026
project_label: Code
---

I competed in the **HackerRank Orchestrate 2026 Hackathon** and built a **conservative, RAG-based support triage agent**. It reads incoming support tickets, retrieves relevant Markdown documentation from the provided corpus, classifies the issue type and product area, and decides whether to answer or escalate. Code is on [GitHub](https://github.com/Rishi-Jain-27/hackerrank-orchestrate-2026).

The headline design choice: I deliberately did *not* let the LLM make routing decisions.

## The Split: Deterministic Routing, LLM-Generated Prose

The agent has two halves, and the dividing line is the most important decision in the system.

**Deterministic rules** handle anything safety-critical:

- **Status** — answered, escalated, or refused.
- **Request type** — what kind of ticket this is.
- **Product area** — which slice of the product the issue belongs to.
- **Escalation** — whether a human needs to look at it.

These don't go anywhere near the model. They're rule-based, auditable, and the same input always produces the same routing output.

**The LLM** only does one thing: write the final response and the justification, grounded entirely in the retrieved snippets the routing layer hands it. It is a writer, not a decider.

This split is the whole reason the agent is trustworthy. Models hallucinate. Models can be jailbroken. Models will confidently route a billing complaint as a documentation question if you give them the chance. Routing has to be deterministic if you want to know, after the fact, why a ticket went where it went.

## Retrieval: Company-Filtered, Path-Aware, Override-Aware

The retrieval layer is doing a lot more than nearest-neighbor over chunks.

- **Company-filtered retrieval.** A ticket from Company A only retrieves against documentation that's actually relevant to Company A. This stops the agent from quoting another tenant's docs at someone.
- **Product-area metadata from paths.** Markdown files live under structured paths, and I extract product-area metadata from those paths on ingest. The doc's filesystem location *is* signal, and ignoring it would have been throwing data away.
- **Nested product-area overrides.** When a deeper folder needs to override its parent's product area (because a sub-section legitimately belongs to a different area), the override is respected. Without this, retrieval drifts subtly wrong on edge cases.

The end result: the snippets the LLM sees are tightly scoped to the right tenant and the right product area before any generation happens.

## Safety Layers

On top of retrieval and rule-based routing, I stacked a series of guards:

- **Prompt-injection handling.** Ticket text is treated as untrusted data, not as instructions. Standard injection attempts (role-overrides, "ignore previous instructions," embedded fake system prompts) are detected and neutralized before the LLM ever sees them.
- **Risk detection.** Anything that hits known risky patterns — security issues, account-specific data requests, credential-shaped strings — gets routed to escalation regardless of how the LLM would have wanted to handle it.
- **Product-area consistency guards.** If the rule-based classifier says one product area and the retrieved docs are predominantly from another, that's a red flag and the agent doesn't try to paper over it — it escalates instead of generating a confident wrong answer.
- **Postprocessing validator.** After the LLM generates its response, the output runs through a validator that checks it stays grounded in the retrieved snippets and consistent with the routing decisions. If the model hallucinates a feature, contradicts a rule, or drifts off-corpus, the validator catches it.

Each of these is individually small. Together they're the difference between a demo and something you'd actually let near a production support queue.

## What It Answers vs. What It Escalates

The agent's behavior is shaped by the same conservatism that drove the architecture:

- It **answers** simple, documented questions where the corpus clearly contains the answer and routing is unambiguous.
- It **escalates** anything that is account-specific, billing, security, assessment-related, an outage, vague, or unsupported by the docs.

That escalation list isn't a failure mode — it's the spec. A triage agent that escalates the right 20% of tickets and answers the other 80% accurately is enormously more valuable than one that tries to answer everything and gets a meaningful slice of it wrong. Confident wrong answers in support are worse than no answers; they erode the human team's trust and create cleanup work.

## Takeaway

The interesting part of building agents right now isn't the model. It's deciding which decisions the model is allowed to make. I gave this one exactly one job — write a response from already-retrieved, already-routed, already-validated context — and built everything around it to make sure that job is the only one it ever does. That's what "conservative" means in this context, and it's why the agent stays grounded.
