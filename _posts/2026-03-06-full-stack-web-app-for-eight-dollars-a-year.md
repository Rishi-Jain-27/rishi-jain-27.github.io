---
layout: post
title: "A full-stack web app for eight dollars a year"
author: "Rishi Jain"
date: 2026-03-06
categories: projects
project_url: https://hwagent.org
project_label: Visit
---

I wanted a small product that I could put online and then mostly forget about, so I built a full-stack web app with a real PostgreSQL backend and a few AI features wired into it, and the whole thing runs for roughly eight dollars a year. It is live at [hwagent.org](https://hwagent.org). The number in the title is the part that surprised me, and most of this post is about why it ended up being so small.

## The stack

The backend is Postgres, the frontend is an ordinary web frontend, and the AI calls sit next to the database rather than in place of it. I reached for Postgres because I already know it and it does exactly what I need, and the rule I held to throughout was that the model only gets involved where having it in the loop is genuinely better than not, with everything else staying as plain CRUD. That decision is what keeps the architecture boring, and boring is what keeps it cheap to run.

## What the AI actually does

The model handles a handful of specific features where a human-quality bit of generation earns its place, and the rest of what you touch is database queries and forms behaving the way you would expect. Keeping the AI confined to those few spots means the rest of the system is predictable, testable, and free, which matters a great deal when the goal is something you can leave running unattended.

## Hardening

Once the features worked I went back and tightened the unglamorous things, which is to say validation on both sides of every form, escaping in every place that user content gets rendered, and no raw HTML ever reaching the DOM. None of that is interesting to build and none of it shows up to a visitor, but it is the part that lets me leave the app live without checking on it, and on a project you intend to ignore for months at a time that quiet robustness is worth more than any feature.

## Where the eight dollars goes

The entire bill is a small managed Postgres instance plus a domain, and nothing else is paid for. The architecture is deliberately tame enough to fit on the free or near-free tier of whatever provider you happen to trust, so the only line items left are the database and the name people type in.

## Takeaway

A real database, a real frontend, a few AI features, and a domain you can hand to someone, for about the price of a single lunch per year. Hosting being cheap is the obvious half of that, and the half worth keeping is the other one: most of what side projects cost is paying for capacity and services you were never actually going to use.
