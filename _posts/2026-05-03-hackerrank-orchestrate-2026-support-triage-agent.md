---
layout: post
title: "HackerRank Orchestrate 2026: a support triage agent that doesn't hallucinate"
author: "Rishi Jain"
date: 2026-05-03 12:00:00 -0500
categories: projects
project_url: https://github.com/Rishi-Jain-27/hackerrank-orchestrate-2026
project_label: Code
---

I competed in the HackerRank Orchestrate 2026 Hackathon and built a conservative, RAG-based support triage agent. It finished #298 out of 1,349 contestants. The agent reads incoming support tickets, retrieves relevant Markdown docs from the provided corpus, classifies the issue type and product area, and decides whether to answer or escalate. Code on [GitHub](https://github.com/Rishi-Jain-27/hackerrank-orchestrate-2026).

The headline design choice: I deliberately did not let the LLM make routing decisions.

## Deterministic routing, LLM-generated prose

The agent has two halves. The split between them is the most important decision in the system.

Deterministic rules handle anything safety-critical. Status (answered, escalated, refused), request type, product area, and the escalation decision are all produced by rule-based logic that the model never sees. The same input always produces the same routing output, and the decision is auditable after the fact.

The LLM does exactly one thing: write the final response and the justification, grounded in the snippets the routing layer hands it. It's a writer, not a decider. Models hallucinate, get jailbroken, and will confidently route a billing complaint as a documentation question if given the chance. Pulling the routing decisions out of the model entirely is the only way to know, after a ticket goes wrong, why it went wrong.

## Retrieval: company-filtered, path-aware, override-aware

The retrieval layer does more than nearest-neighbor over chunks.

- **Company-filtered retrieval.** A ticket from Company A only retrieves against documentation relevant to Company A. The agent never quotes another tenant's docs at someone.
- **Product-area metadata from paths.** Markdown files live under structured paths, and I extract product-area metadata from those paths on ingest. The location of a file in the tree carries real signal; ignoring it would have been throwing data away.
- **Nested overrides.** When a deeper folder needs to override its parent's product area because a subsection legitimately belongs elsewhere, the override wins. Without that, retrieval drifts wrong on edge cases.

The snippets the LLM sees are tightly scoped to the right tenant and the right product area before any generation happens.

## Safety layers

On top of retrieval and rule-based routing I stacked a few guards:

- **Prompt-injection handling.** Ticket text is treated as untrusted data, not as instructions. Role-overrides, "ignore previous instructions," embedded fake system prompts, the standard tricks all get detected and neutralized before the model ever sees them.
- **Risk detection.** Known risky patterns like security issues, account-specific data requests, credential-shaped strings all route to escalation regardless of what the LLM would have done.
- **Product-area consistency.** If the rule-based classifier says one product area and the retrieved docs are predominantly from another, the agent escalates instead of papering over the conflict.
- **Postprocessing validator.** After the LLM writes its response, a validator checks that the output stays grounded in the retrieved snippets and consistent with the routing decisions. Hallucinated features, contradictions, drift off-corpus, all caught here.

Individually each guard is small. Stacked together they're what separates a demo from something I'd be willing to put near a real support queue.

## What it answers vs. what it escalates

The behavior follows the same conservatism as the architecture. The agent answers simple, documented questions where the corpus clearly contains the answer and routing is unambiguous. It escalates anything account-specific, billing, security, assessment-related, an outage, vague, or unsupported by the docs.

That escalation list isn't a failure mode, it's the spec. A triage agent that escalates the right 20% of tickets and answers the other 80% correctly is much more useful than one that tries to answer everything and gets a meaningful chunk wrong. Confident wrong answers erode the human team's trust in the agent's output and create cleanup work that didn't exist before.

## Takeaway

The interesting part of building agents right now isn't the model itself. It's deciding which decisions the model is allowed to make. I gave this one a single job, writing a response from already-retrieved, already-routed, already-validated context, and built everything around it to keep it inside that job.
